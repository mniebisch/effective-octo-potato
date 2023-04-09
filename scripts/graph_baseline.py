import json
import pathlib
from typing import Any, Dict, List, Tuple

import numpy as np
import numpy.typing as npt

# import onnx
# import onnxruntime
import pandas as pd
import torch
import torch_geometric.data as pyg_data
import torch_geometric.loader as pyg_loader
import torch_geometric.nn as pyg_nn
import tqdm
from sklearn import model_selection

from effective_octo_potato.graph_utils import (
    apply_node_mask_to_edges,
    calc_node_dist_to_reference_feature,
    compute_reference_nodes,
    create_edge_index,
    create_node_indices,
    create_node_mask,
    create_one_hot,
)

# TODO 1 up-/or downsampling of frames to fixed number of frames via interpolation [IMPLEMENTED]
# TODO 2 could a kinda transformer handle variable size of frames?
# TODO 3 create huge graph reflecting temporal patterns (not compatible with TODO 2) [IMPLEMENTED]
# TODO 4 use/create edge labels for temporal pattern graph in TODO 3


def load_relevant_data_subset(pq_path: pathlib.Path) -> npt.NDArray[np.float32]:
    ROWS_PER_FRAME = 543  # number of landmarks per frame
    data_columns = ["x", "y", "z"]
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)


def _create_features(
    file_name: pathlib.Path, feature_generator: torch.nn.Module
) -> Tuple[torch.Tensor, torch.Tensor]:
    sign_sequence = load_relevant_data_subset(pq_path=file_name)
    sign_sequence = torch.from_numpy(sign_sequence)
    features, edge_index = feature_generator(sign_sequence)
    return features, edge_index


def create_features(
    file_names: List[pathlib.Path], feature_generator: torch.nn.Module
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    return [
        _create_features(filename, feature_generator)
        for filename in tqdm.tqdm(file_names)
    ]


def handle_training_data(
    raw_dir: pathlib.Path,
    feature_dir: pathlib.Path,
    feature_generator: torch.nn.Module,
    feature_file_name: str,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    feature_matrix_file_name = feature_dir / feature_file_name
    if not feature_matrix_file_name.is_file():
        # TODO fix use of data from outer scope
        file_paths = [raw_dir / filename for filename in train_df["path"].tolist()]
        feature_matrix = create_features(file_paths, feature_generator)
        torch.save(feature_matrix, feature_matrix_file_name)
    else:
        feature_matrix = torch.load(feature_matrix_file_name)
    return feature_matrix


class FeatureGenerator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.node_indices: torch.Tensor = create_node_indices()
        self.node_mask: torch.Tensor = create_node_mask(node_indices=self.node_indices)
        self.edge_index: torch.Tensor = create_edge_index()
        # TODO add indices of desired nodes (absolute as we have face atm)
        self.edge_index = apply_node_mask_to_edges(
            mask=self.node_mask, edge_index=self.edge_index
        )
        self.one_hot = create_one_hot(num_nodes=self.node_indices.shape[0])

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
        face_x = x[:, :468].contiguous().view(-1, 468 * 3)
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
        face_mean = torch.mean(face_x, 0)
        face_mean = face_mean.view(468, 3)

        lefth_std = torch.std(lefth_x, 0)
        lefth_std = lefth_std.view(21, 3)
        righth_std = torch.std(righth_x, 0)
        righth_std = righth_std.view(21, 3)
        pose_std = torch.std(pose_x, 0)
        pose_std = pose_std.view(33, 3)
        face_std = torch.std(face_x, 0)
        face_std = face_std.view(468, 3)

        lefth_feat = torch.cat([lefth_mean, lefth_std], dim=1)
        righth_feat = torch.cat([righth_mean, righth_std], dim=1)
        pose_feat = torch.cat([pose_mean, pose_std], dim=1)
        face_feat = torch.cat([face_mean, face_std], dim=1)

        x_feat = torch.cat([face_feat, lefth_feat, pose_feat, righth_feat], dim=0)
        # compute_reference_nodes needs to be applied to all 543 nodes!
        reference_coords = compute_reference_nodes(x_feat[:, [0, 2, 4]])
        x_feat = x_feat[self.node_indices]

        nan_mask = torch.isnan(x_feat)
        nan_mask = torch.any(nan_mask, dim=1)
        nan_mask = torch.logical_not(nan_mask)

        x_feat = x_feat[nan_mask]
        # ATTENTION indexing due to mean and std stuff
        # Further keep in mind that we would like to train for 2D case? maybe
        reference_feat = calc_node_dist_to_reference_feature(
            nodes=x_feat[:, [0, 2, 4]], reference=reference_coords
        )

        one_hot = self.one_hot[nan_mask]
        edge_index = self.edge_index
        edge_index = apply_node_mask_to_edges(mask=nan_mask, edge_index=edge_index)

        # TEMPORARY [END]
        return torch.cat([x_feat, one_hot, reference_feat], dim=1), edge_index


def _get_label_map(data_dir: pathlib.Path) -> Dict[str, int]:
    label_csv = "sign_to_prediction_index_map.json"
    with open(data_dir / label_csv) as file:
        label_map = json.load(file)
    return label_map


def load_labels(data_dir: pathlib.Path, labels: pd.Series) -> npt.NDArray[np.integer]:
    label_map = _get_label_map(data_dir=data_dir)
    labels = labels.replace(label_map)
    return labels.values


class MeinBlock(torch.nn.Module):
    def __init__(
        self, input_channels: int, hidden_channels: int, aggr: str = "sum"
    ) -> None:
        super().__init__()
        self.act1 = torch.nn.PReLU()
        self.bn1 = torch.nn.BatchNorm1d(input_channels)
        self.conv1 = pyg_nn.GCNConv(input_channels, hidden_channels, aggr=aggr)
        self.act2 = torch.nn.PReLU()
        self.conv2 = pyg_nn.GCNConv(hidden_channels, input_channels, aggr=aggr)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.act1(x)
        x = self.bn1(x)
        x = self.conv1(x, edge_index)
        x = self.act2(x)
        x = self.conv2(x, edge_index)
        return x


class GCN(torch.nn.Module):
    def __init__(
        self,
        num_node_features: int,
        hidden_channels: int,
        num_classes: int,
        num_blocks: int,
        drop_p: float,
    ) -> None:
        super().__init__()

        self.pre_fc1 = torch.nn.Linear(num_node_features, hidden_channels)
        self.blocks = torch.nn.ModuleList(
            [MeinBlock(hidden_channels, 2 * hidden_channels) for _ in range(num_blocks)]
        )
        self.post_bn = torch.nn.BatchNorm1d(hidden_channels)
        self.post_fc1 = torch.nn.Linear(hidden_channels, hidden_channels * 4)
        self.post_act1 = torch.nn.PReLU()
        self.dropout = torch.nn.Dropout(p=drop_p)
        self.post_fc2 = torch.nn.Linear(hidden_channels * 4, num_classes)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        x = self.pre_fc1(x)
        for i in range(len(self.blocks)):
            x = self.blocks[i](x, edge_index)
        x = pyg_nn.global_mean_pool(x, batch)
        x = self.post_bn(x)
        x = self.post_fc1(x)
        x = self.post_act1(x)
        x = self.dropout(x)
        x = self.post_fc2(x)
        return x


def get_data_split(
    group_labels: pd.Series, train_size: float = 0.8, is_submission: bool = True
) -> Tuple[npt.NDArray[np.integer], npt.NDArray[np.integer]]:
    if is_submission:
        raise NotImplementedError

    splitter = model_selection.GroupShuffleSplit(n_splits=1, train_size=train_size)
    group_split = splitter.split(None, None, group_labels)
    return next(group_split)


def eval(
    model: torch.nn.Module,
    dataloader: pyg_loader.DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        batch_iterator = tqdm.tqdm(dataloader)
        for data in batch_iterator:
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            batch = data.batch.to(device)
            prediction = model(x, edge_index, batch)
            target = torch.tensor(data.y, dtype=torch.long)
            prediction_label = torch.argmax(prediction, dim=1)
            prediction_label = prediction_label.detach().cpu()
            total += target.shape[0]
            correct += (target == prediction_label).sum().item()
    return correct / total


if __name__ == "__main__":
    data_base_path = pathlib.Path(__file__).parent.parent / "data"
    output_base_path = pathlib.Path(__file__).parent.parent / "data"

    is_submission = False

    # load/process labels
    data_csv = "train.csv"
    train_df = pd.read_csv(data_base_path / data_csv)
    labels = load_labels(data_dir=data_base_path, labels=train_df["sign"])

    # create/load features
    feature_matrix_file_name = "graph_data_meanstd_subpose_reference_nodes.zip"
    fg = FeatureGenerator()

    feature_matrix = handle_training_data(
        raw_dir=data_base_path,
        feature_dir=output_base_path,
        feature_generator=fg,
        feature_file_name=feature_matrix_file_name,
    )

    graphs: List[pyg_data.Data] = [
        pyg_data.Data(x=node_features, edge_index=edge_index, y=label)
        for (node_features, edge_index), label in zip(feature_matrix, labels)
    ]

    # split data
    train_indices, valid_indices = get_data_split(
        group_labels=train_df["participant_id"], is_submission=is_submission
    )
    train_indices = train_indices.tolist()
    valid_indices = valid_indices.tolist()

    def _indexing_helper(data: List[Any], indices: List[int]) -> list[Any]:
        return [data[ind] for ind in indices]

    train_graphs = _indexing_helper(graphs, train_indices)
    valid_graphs = _indexing_helper(graphs, valid_indices)

    # hyperparams
    batch_size = 128
    epochs = 1  # 175

    # pyg stuff
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dataloader = pyg_loader.DataLoader(
        train_graphs, batch_size=batch_size, shuffle=True
    )
    model = GCN(
        num_node_features=graphs[0].num_node_features,
        hidden_channels=64,
        num_classes=250,
        num_blocks=1,
        drop_p=0.0,
    )
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5 * 10e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=25,
        T_mult=2,
        verbose=True,
    )

    model.train()
    for epoch_idx in range(epochs):
        batch_iterator = tqdm.tqdm(
            train_dataloader, desc=f"Epoch: {epoch_idx+1:02d}/{epochs}"
        )
        rolling_loss = None
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
            if rolling_loss is None:
                rolling_loss = loss.item()
            else:
                rolling_loss = 0.9 * rolling_loss + 0.1 * loss.item()
            batch_iterator.set_postfix({"loss": rolling_loss})
        scheduler.step()

    # eval
    valid_dataloader = pyg_loader.DataLoader(
        valid_graphs, batch_size=batch_size, shuffle=True
    )

    train_acc = eval(model, train_dataloader, device)
    valid_acc = eval(model, valid_dataloader, device)
    print("train acc", train_acc)
    print("valid acc", valid_acc)

    # # onnx check
    # # # shapes
    # # x shape [num_nodes * batch_size, num_node_features]
    # # edge_index shape [2, batch_size * num_edges]
    # # batch shape [num_nodes * batch_size]
    # # model = torch_geometric.compile(model)
    # num_nodes_in = graphs[0].num_nodes
    # num_node_features_in = graphs[0].num_node_features
    # num_edges = graphs[0].num_edges

    # # # create input dummies
    # x_dummy = torch.randn(
    #     num_nodes_in * batch_size,
    #     num_node_features_in,
    #     dtype=torch.float32,
    #     requires_grad=True,
    # )
    # # 9856
    # edge_index_dummy = torch.tensor(
    #     [(0, 1), (1, 0)] * (77 * batch_size), dtype=torch.int64
    # ).T
    # batch_dummy = torch.arange(batch_size, dtype=torch.int64)
    # batch_dummy: torch.Tensor = batch_dummy.repeat_interleave(num_nodes_in)

    # x_dummy = x_dummy.to(device)
    # edge_index_dummy = edge_index_dummy.to(device)
    # batch_dummy = batch_dummy.to(device)
    # # # perform onnx conversion
    # model.eval()
    # torch.onnx.export(
    #     model,
    #     (x_dummy, edge_index_dummy, batch_dummy),
    #     "graph_model.onnx",
    #     export_params=True,
    #     opset_version=16,
    #     do_constant_folding=True,
    #     input_names=["features", "edges", "batch"],
    #     output_names=["output"],
    #     dynamic_axes={
    #         "features": [0],
    #         "edges": [1],
    #         "batch": [0],
    #         "output": [0],
    #     },
    # )

    # onnx_model = onnx.load("graph_model.onnx")
    # # raises Exception if something went wrong
    # check = onnx.checker.check_model(onnx_model)

    # # compare values
    # output_dummy = model(x_dummy, edge_index_dummy, batch_dummy)

    # def to_numpy(tensor):
    #     return (
    #         tensor.detach().cpu().numpy()
    #         if tensor.requires_grad
    #         else tensor.cpu().numpy()
    #     )

    # ort_session = onnxruntime.InferenceSession("graph_model.onnx")
    # ort_inputs = {
    #     ort_session.get_inputs()[0].name: to_numpy(x_dummy),
    #     ort_session.get_inputs()[1].name: to_numpy(edge_index_dummy),
    #     ort_session.get_inputs()[2].name: to_numpy(batch_dummy),
    # }
    # ort_outs = ort_session.run(None, ort_inputs)

    # np.testing.assert_allclose(
    #     to_numpy(output_dummy), ort_outs[0], rtol=1e-03, atol=1e-05
    # )
