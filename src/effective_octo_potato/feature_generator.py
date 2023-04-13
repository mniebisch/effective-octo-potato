import pathlib
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch_geometric.data as pyg_data
import tqdm

from effective_octo_potato.graph_utils import (
    apply_node_mask_to_edges,
    calc_node_dist_to_reference_feature,
    compute_reference_nodes,
    create_edge_index,
    create_node_indices,
    create_node_mask,
    create_one_hot,
)
from effective_octo_potato.kaggle import load_relevant_data_subset

__all__ = ["FeatureGenerator", "create_pyg_dataset", "handle_training_data"]


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
        self.one_hot: torch.Tensor = create_one_hot(
            num_nodes=self.node_indices.shape[0]
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


def _create_features(
    file_name: pathlib.Path, feature_generator: FeatureGenerator
) -> Tuple[torch.Tensor, torch.Tensor]:
    sign_sequence = load_relevant_data_subset(pq_path=file_name)
    sign_sequence = torch.from_numpy(sign_sequence)
    features, edge_index = feature_generator(sign_sequence)
    return features, edge_index


def create_features(
    file_names: List[pathlib.Path], feature_generator: FeatureGenerator
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    return [
        _create_features(filename, feature_generator)
        for filename in tqdm.tqdm(file_names)
    ]


def handle_training_data(
    data: pd.DataFrame,
    raw_dir: pathlib.Path,
    feature_dir: pathlib.Path,
    feature_generator: FeatureGenerator,
    feature_file_name: str,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    feature_matrix_file_name = feature_dir / feature_file_name
    if not feature_matrix_file_name.is_file():
        # TODO fix use of data from outer scope
        file_paths = [raw_dir / filename for filename in data["path"].tolist()]
        feature_matrix = create_features(file_paths, feature_generator)
        torch.save(feature_matrix, feature_matrix_file_name)
    else:
        feature_matrix = torch.load(feature_matrix_file_name)
    return feature_matrix


def create_pyg_dataset(
    feature_matrix: List[
        Tuple[
            torch.Tensor,
            torch.Tensor,
        ]
    ],
    labels: npt.NDArray[np.integer],
) -> List[pyg_data.Data]:
    return [
        pyg_data.Data(
            x=node_features,
            edge_index=edge_index,
            y=torch.tensor(label),
        )
        for (node_features, edge_index), label in zip(feature_matrix, labels)
    ]
