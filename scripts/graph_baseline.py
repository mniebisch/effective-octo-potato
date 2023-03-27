import collections
import json
import pathlib
from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt
import onnx
import onnxruntime
import pandas as pd
import torch
import torch.nn.functional as torch_F
import torch_geometric.data as pyg_data
import torch_geometric.loader as pyg_loader
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
import tqdm

from effective_octo_potato.graph_utils import (
    apply_node_mask_to_edges,
    create_edge_index,
    create_node_indices,
    create_node_mask,
)

# TODO get edge based on media type descriptions
# TODO create multi-graph dataset similar to TUDataset (Mutag) (pyg)
# https://colab.research.google.com/drive/1I8a0DfQ3fI7Njc62__mVXUlcAleUclnb?usp=sharing#scrollTo=j11WiUr-PRH_


# https://stackoverflow.com/questions/53033556/how-should-the-learning-rate-change-as-the-batch-size-change

# https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html

# TODO Check if pyg can be transformed to tflight
# for more check doc


def load_relevant_data_subset(pq_path: pathlib.Path) -> npt.NDArray[np.float32]:
    ROWS_PER_FRAME = 543  # number of landmarks per frame
    data_columns = ["x", "y", "z"]
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)


class FeatureGenerator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.node_indices: torch.Tensor = create_node_indices()
        self.node_mask: torch.Tensor = create_node_mask()
        self.edge_index: torch.Tensor = create_edge_index()
        self.edge_index = apply_node_mask_to_edges(
            mask=self.node_mask, edge_index=self.edge_index
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Feature Transformation.
        Args:
            x: Sign sequence with shape:
               [num frames, num landmarks, num spatial dims]
               [num frames, 543, 3] of datatype float32.
        Returns:
            Feature vector representing sign sequence with shape
            [num selected landmarks * num spatial coordinates  * num features].
            Eg. num landmarks == 85 if only left hand, right hand, and pose are
            used.
        """
        # TEMPORARY TO GET EVERYTHING RUNNING [START]
        lefth_x = x[:, 468:489, :].contiguous().view(-1, 21 * 3)
        pose_x = x[:, 489:522, :].contiguous().view(-1, 33 * 3)
        righth_x = x[:, 522:, :].contiguous().view(-1, 21 * 3)

        lefth_x = lefth_x[~torch.any(torch.isnan(lefth_x), dim=1), :]
        righth_x = righth_x[~torch.any(torch.isnan(righth_x), dim=1), :]

        lefth_mean = torch.mean(lefth_x, 0)
        lefth_mean = lefth_mean.view(21, 3)
        righth_mean = torch.mean(righth_x, 0)
        righth_mean = righth_mean.view(21, 3)
        pose_mean = torch.mean(pose_x, 0)
        pose_mean = pose_mean.view(33, 3)

        lefth_std = torch.std(lefth_x, 0)
        lefth_std = lefth_std.view(21, 3)
        righth_std = torch.std(righth_x, 0)
        righth_std = righth_std.view(21, 3)
        pose_std = torch.std(pose_x, 0)
        pose_std = pose_std.view(33, 3)

        lefth_feat = torch.cat([lefth_mean, lefth_std], dim=1)
        righth_feat = torch.cat([righth_mean, righth_std], dim=1)
        pose_feat = torch.cat([pose_mean, pose_std], dim=1)

        x_feat = torch.cat([lefth_feat, pose_feat, righth_feat], dim=0)

        # TODO one hot encoding
        # TODO drop nans
        nan_mask = torch.isnan(x_feat)
        x_feat = torch.where(nan_mask, torch.tensor(0.0, dtype=torch.float32), x_feat)
        edge_index = self.edge_index

        # TEMPORARY [END]
        return x_feat, edge_index


def _get_label_map(data_dir: pathlib.Path) -> Dict[str, int]:
    label_csv = "sign_to_prediction_index_map.json"
    with open(data_dir / label_csv) as file:
        label_map = json.load(file)
    return label_map


def load_labels(data_dir: pathlib.Path, labels: pd.Series) -> npt.NDArray[np.integer]:
    label_map = _get_label_map(data_dir=data_dir)
    labels = labels.replace(label_map)
    return labels.values


def create_pyg_graph(
    node_features: npt.NDArray[np.float32], edge_matrix: torch.Tensor, label: int
) -> pyg_data.Data:
    node_features = node_features.reshape((75, 6))
    node_features = torch.tensor(node_features, dtype=torch.float32)
    return pyg_data.Data(x=node_features, edge_index=edge_matrix, y=label)


class GCN(torch.nn.Module):
    def __init__(
        self, num_node_features: int, hidden_channels: int, num_classes: int
    ) -> None:
        super().__init__()
        self.conv1 = pyg_nn.GCNConv(num_node_features, hidden_channels)
        self.conv2 = pyg_nn.GCNConv(hidden_channels, hidden_channels)
        self.conv3 = pyg_nn.GCNConv(hidden_channels, hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, num_classes)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.nn.functional.relu(x)
        x = self.conv3(x, edge_index)
        x = pyg_nn.global_mean_pool(x, batch)
        x = self.linear(x)
        return x


if __name__ == "__main__":
    data_base_path = pathlib.Path(__file__).parent.parent / "data"

    # load/process labels
    data_csv = "train.csv"
    train_df = pd.read_csv(data_base_path / data_csv)
    labels = load_labels(data_dir=data_base_path, labels=train_df["sign"])

    fg = FeatureGenerator()
    oi = load_relevant_data_subset(data_base_path / train_df["path"][0])
    oi = torch.from_numpy(oi)
    blub = fg(oi)

    example_data = pd.read_parquet(data_base_path / train_df["path"][0])

    num_nodes = 543
    nodes: pd.DataFrame = example_data[["type", "landmark_index"]].iloc[
        range(num_nodes)
    ]

    # nodes from view of kaggle data (DATASET PERSPECTIVE)
    left_hand = list(range(468, 489))
    right_hand = list(range(522, 543))
    pose = list(range(489, 522))
    node_indices = left_hand + right_hand + pose

    # node selection
    nodes = nodes.iloc[node_indices]
    node_map: dict[str, dict[int, int]] = collections.defaultdict(dict)
    for feature_matrix_index, (body_part, landmark_index) in enumerate(
        zip(nodes.type, nodes.landmark_index)
    ):
        node_map[body_part][landmark_index] = feature_matrix_index

    # edges from view of body part (BODY PART PERSPECTIVE)
    hand_edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (5, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (9, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (13, 17),
        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),
    ]

    pose_edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 7),
        (0, 4),
        (4, 5),
        (5, 6),
        (6, 8),
        (9, 10),
        (11, 12),
        (11, 13),
        (11, 23),
        (13, 15),
        (15, 21),
        (15, 19),
        (15, 17),
        (17, 19),
        (12, 14),
        (12, 24),
        (14, 16),
        (16, 22),
        (16, 18),
        (16, 20),
        (18, 20),
        (23, 24),
        (23, 25),
        (25, 27),
        (27, 29),
        (27, 31),
        (29, 31),
        (24, 26),
        (26, 28),
        (28, 30),
        (28, 32),
        (30, 32),
    ]

    body_part_edges = {
        "left_hand": hand_edges,
        "right_hand": hand_edges,
        "pose": pose_edges,
    }

    graph_edges = [
        (node_map[body_part][edge_x], node_map[body_part][edge_y])
        for body_part, edges in body_part_edges.items()
        for edge_x, edge_y in edges
    ]

    # shape feature matrix
    # (76139 X 75 * 3 * 2)
    # (num_graph X num_nodes * num_spatial_coords * num_features)
    # num_features = 2 => mean and std
    feature_matrix_file_name = "baseline_mean_std.npy"
    feature_matrix = np.load(data_base_path / feature_matrix_file_name)

    example_graph = feature_matrix[0]

    edge_matrix = torch.tensor(graph_edges, dtype=torch.int64).T
    edge_matrix = pyg_utils.to_undirected(edge_matrix)

    graphs: List[pyg_data.Data] = [
        create_pyg_graph(node_features, edge_matrix, label)
        for node_features, label in tqdm.tqdm(zip(feature_matrix, labels))
    ]

    # hyperparams
    batch_size = 64
    epochs = 1

    # pyg stuff
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dataloader = pyg_loader.DataLoader(
        graphs, batch_size=batch_size, shuffle=True
    )
    model = GCN(
        num_node_features=graphs[0].num_node_features,
        hidden_channels=64,
        num_classes=250,
    )
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=5 * 10e-2)

    model.train()
    for epoch_idx in range(epochs):
        batch_iterator = tqdm.tqdm(
            train_dataloader, desc=f"Epoch: {epoch_idx+1:02d}/{epochs}"
        )
        for data in batch_iterator:
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            batch = data.batch.to(device)
            optimizer.zero_grad()
            prediction = model(x, edge_index, batch)
            target = torch.tensor(data.y, dtype=torch.long)
            target = target.to(device)
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()
            batch_iterator.set_postfix({"loss": loss.item()})

    # onnx check
    # # shapes
    # x shape [num_nodes * batch_size, num_node_features]
    # edge_index shape [2, batch_size * num_edges]
    # batch shape [num_nodes * batch_size]
    # model = torch_geometric.compile(model)
    num_nodes_in = graphs[0].num_nodes
    num_node_features_in = graphs[0].num_node_features
    num_edges = graphs[0].num_edges

    # # create input dummies
    x_dummy = torch.randn(
        num_nodes_in * batch_size,
        num_node_features_in,
        dtype=torch.float32,
        requires_grad=True,
    )
    # 9856
    edge_index_dummy = torch.tensor(
        [(0, 1), (1, 0)] * (77 * batch_size), dtype=torch.int64
    ).T
    batch_dummy = torch.arange(batch_size, dtype=torch.int64)
    batch_dummy: torch.Tensor = batch_dummy.repeat_interleave(num_nodes_in)

    x_dummy = x_dummy.to(device)
    edge_index_dummy = edge_index_dummy.to(device)
    batch_dummy = batch_dummy.to(device)
    # # perform onnx conversion
    model.eval()
    # model = torch.jit.script(model)
    # model = torch.jit.trace(model)
    torch.onnx.export(
        model,
        (x_dummy, edge_index_dummy, batch_dummy),
        "graph_model.onnx",
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=["features", "edges", "batch"],
        output_names=["output"],
        dynamic_axes={
            "features": [0],
            "edges": [1],
            "batch": [0],
            "output": [0],
        },
    )

    onnx_model = onnx.load("graph_model.onnx")
    # raises Exception if something went wrong
    check = onnx.checker.check_model(onnx_model)

    # compare values
    output_dummy = model(x_dummy, edge_index_dummy, batch_dummy)

    def to_numpy(tensor):
        return (
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            else tensor.cpu().numpy()
        )

    ort_session = onnxruntime.InferenceSession("graph_model.onnx")
    ort_inputs = {
        ort_session.get_inputs()[0].name: to_numpy(x_dummy),
        ort_session.get_inputs()[1].name: to_numpy(edge_index_dummy),
        ort_session.get_inputs()[2].name: to_numpy(batch_dummy),
    }
    ort_outs = ort_session.run(None, ort_inputs)

    np.testing.assert_allclose(
        to_numpy(output_dummy), ort_outs[0], rtol=1e-03, atol=1e-05
    )
