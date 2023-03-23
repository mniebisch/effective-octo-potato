import collections
import json
import pathlib
from typing import Dict, List

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn.functional as torch_F
import torch_geometric.data as pyg_data
import torch_geometric.loader as pyg_loader
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
import tqdm
from torch_geometric.nn import GCNConv

# TODO get edge based on media type descriptions
# TODO create multi-graph dataset similar to TUDataset (Mutag) (pyg)
# https://colab.research.google.com/drive/1I8a0DfQ3fI7Njc62__mVXUlcAleUclnb?usp=sharing#scrollTo=j11WiUr-PRH_


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

    edge_matrix = torch.tensor(graph_edges, dtype=torch.int).T
    edge_matrix = pyg_utils.to_undirected(edge_matrix)

    graphs: List[pyg_data.Data] = [
        create_pyg_graph(node_features, edge_matrix, label)
        for node_features, label in tqdm.tqdm(zip(feature_matrix, labels))
    ]

    # pyg stuff
    loader = pyg_loader.DataLoader(graphs, batch_size=4)
    model = GCN(
        num_node_features=graphs[0].num_node_features,
        hidden_channels=64,
        num_classes=250,
    )

    criterion = torch.nn.CrossEntropyLoss()

    for data in loader:
        prediction = model(data.x, data.edge_index, data.batch)
        target = torch.tensor(data.y, dtype=torch.long)
        loss = criterion(prediction, target)

    print("oi")
