import collections
import pathlib

import numpy as np
import pandas as pd
import torch
import torch_geometric.data as pyg_data
import torch_geometric.utils as pyg_utils

# TODO get edge based on media type descriptions
# TODO create multi-graph dataset similar to TUDataset (Mutag) (pyg)
# https://colab.research.google.com/drive/1I8a0DfQ3fI7Njc62__mVXUlcAleUclnb?usp=sharing#scrollTo=j11WiUr-PRH_

if __name__ == "__main__":
    data_base_path = pathlib.Path(__file__).parent.parent / "data"

    # load/process labels
    data_csv = "train.csv"
    train_df = pd.read_csv(data_base_path / data_csv)

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
    example_graph = example_graph.reshape((75, 6))

    edge_matrix = torch.tensor(graph_edges, dtype=torch.int).T

    sign_example = pyg_data.Data(
        x=torch.tensor(example_graph, dtype=torch.float32),
        edge_index=pyg_utils.to_undirected(edge_matrix),
    )

    print(example_data.head())
