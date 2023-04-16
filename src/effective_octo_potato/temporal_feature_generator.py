import pathlib
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch_geometric.data as pyg_data
import torch_geometric.transforms as pyg_transforms
import tqdm

from effective_octo_potato.graph_utils import (
    apply_node_mask_to_edges,
    compute_reference_nodes,
    create_edge_index,
    create_node_indices,
    create_node_mask,
    create_one_hot,
    create_temporal_edge_indices,
    fix_num_frames,
    map_edge_indices_to_temporal_graph,
)
from effective_octo_potato.kaggle import load_relevant_data_subset

__all__ = [
    "GraphDatset",
    "TemporalFeatureGenerator",
    "create_pyg_dataset",
    "handle_training_data",
]


GraphDescription = Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


class TemporalFeatureGenerator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.node_indices: torch.Tensor = create_node_indices()
        self.node_mask: torch.Tensor = create_node_mask(node_indices=self.node_indices)
        self.edge_index: torch.Tensor = create_edge_index()
        self.edge_index = apply_node_mask_to_edges(
            mask=self.node_mask, edge_index=self.edge_index
        )
        # TODO maybe change due to use of node information
        self.one_hot: torch.Tensor = create_one_hot(
            num_nodes=self.node_indices.shape[0]
        )
        self.num_time_steps: int = 5

    def forward(self, x: torch.Tensor) -> GraphDescription:
        """Feature Transformation.
        Args:
            x: Sign sequence with shape:
               [num frames, num landmarks, num spatial dims]
               [num frames, 543, 3] of datatype float32.
        Returns:
            - Node features.
            - COO adjencency matrix.
            - Node indices.
            - Time step.
        """
        # x = x[:, self.node_indices, :] # TODO would be speedup
        x = fix_num_frames(x=x, num_sampling_time_steps=self.num_time_steps)

        node_mask_frame_wise = torch.zeros(
            (self.num_time_steps, len(self.node_indices)), dtype=torch.bool
        )
        node_indices_frame_wise = self.node_indices.repeat(self.num_time_steps, 1)
        time_steps_frame_wise = torch.arange(self.num_time_steps).repeat(
            len(self.node_indices), 1
        )
        time_steps_frame_wise = time_steps_frame_wise.T
        num_refernce_nodes = compute_reference_nodes(torch.zeros(543, 3)).shape[0]
        reference_nodes_frame_wise = torch.zeros(
            (self.num_time_steps, num_refernce_nodes, 3), dtype=torch.float32
        )
        node_features_frame_wise = torch.zeros(
            (self.num_time_steps, len(self.node_indices), 3), dtype=torch.float32
        )
        one_hot_frame_wise = torch.zeros(
            (self.num_time_steps, self.one_hot.shape[0], self.one_hot.shape[1]),
            dtype=torch.float32,
        )
        for time_ind in range(self.num_time_steps):
            # node features
            node_feat_time = x[time_ind]

            # # reference features
            # compute_reference_nodes needs to be applied to all 543 nodes!
            reference_coords = compute_reference_nodes(node_feat_time)
            node_feat_time = node_feat_time[self.node_indices]
            reference_nodes_frame_wise[time_ind] = reference_coords
            one_hot_frame_wise[time_ind] = self.one_hot

            # # filter nan
            nan_mask = torch.isnan(node_feat_time)
            nan_mask = torch.any(nan_mask, dim=1)
            nan_mask = torch.logical_not(nan_mask)
            node_mask_frame_wise[time_ind] = nan_mask

            node_features_frame_wise[time_ind] = node_feat_time

        # create edge indices
        edge_index_frame_wise = map_edge_indices_to_temporal_graph(
            edge_index=self.edge_index,
            num_nodes=len(self.node_indices),
            num_frames=self.num_time_steps,
        )
        # TODO only torch solution instead of list comprehension?
        edge_index = torch.cat(
            [edge_index_time_step for edge_index_time_step in edge_index_frame_wise],
            dim=1,
        )
        edge_index_temporal = create_temporal_edge_indices(
            num_nodes=len(self.node_indices), num_frames=self.num_time_steps
        )
        edge_index = torch.cat([edge_index, edge_index_temporal], dim=1)
        edge_index = apply_node_mask_to_edges(
            mask=node_mask_frame_wise.reshape(-1), edge_index=edge_index
        )

        def _apply_nan_filter(
            nan_masks_frame_wise: torch.Tensor, data_frame_wise: torch.Tensor
        ) -> torch.Tensor:
            """
            Apply mask to filter nans to each time step and stack nodes of separate
            time steps into a single joint node representation.
            """
            return torch.cat(
                [
                    frame_data[frame_mask]
                    for frame_mask, frame_data in zip(
                        nan_masks_frame_wise, data_frame_wise
                    )
                ],
                dim=0,
            )

        # filter nans from node information
        node_features = _apply_nan_filter(
            node_mask_frame_wise, node_features_frame_wise
        )
        node_xyz = node_features
        node_indices = _apply_nan_filter(node_mask_frame_wise, node_indices_frame_wise)
        time_steps = _apply_nan_filter(node_mask_frame_wise, time_steps_frame_wise)
        node_time_steps = time_steps

        # reference information from pose nodes which are never NaN
        reference_features = torch.cat(
            [frame_data for frame_data in reference_nodes_frame_wise], dim=0
        )
        reference_xyz = reference_features
        reference_time_steps = torch.arange(num_refernce_nodes).repeat_interleave(
            self.num_time_steps
        )

        one_hot = _apply_nan_filter(node_mask_frame_wise, one_hot_frame_wise)

        return (
            node_xyz,
            node_indices,
            node_time_steps,
            edge_index,
            reference_xyz,
            reference_time_steps,
            one_hot,
        )


def _create_features(
    file_name: pathlib.Path, feature_generator: TemporalFeatureGenerator
) -> GraphDescription:
    sign_sequence = load_relevant_data_subset(pq_path=file_name)
    sign_sequence = torch.from_numpy(sign_sequence)
    return feature_generator(sign_sequence)


def create_features(
    file_names: List[pathlib.Path], feature_generator: TemporalFeatureGenerator
) -> List[GraphDescription]:
    return [
        _create_features(filename, feature_generator)
        for filename in tqdm.tqdm(file_names)
    ]


def handle_training_data(
    data: pd.DataFrame,
    raw_dir: pathlib.Path,
    feature_dir: pathlib.Path,
    feature_generator: TemporalFeatureGenerator,
    feature_file_name: str,
) -> List[GraphDescription]:
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
    feature_matrix: List[GraphDescription],
    labels: npt.NDArray[np.integer],
    transform: Optional[pyg_transforms.BaseTransform] = None,
) -> pyg_data.InMemoryDataset:
    return GraphDataset(data=feature_matrix, labels=labels, transform=transform)


class GraphDataset(pyg_data.InMemoryDataset):
    def __init__(
        self,
        data: List[GraphDescription],
        labels: npt.NDArray[np.integer],
        transform: Optional[pyg_transforms.BaseTransform] = None,
        pre_transform: Optional[pyg_transforms.BaseTransform] = None,
    ) -> None:
        super().__init__(None, transform, pre_transform)
        self.graph_information = data
        self.labels = labels

    def len(self) -> int:
        return len(self.graph_information)

    def get(self, idx: int) -> pyg_data.Data:
        (
            node_xyz,
            node_indices,
            node_time_steps,
            edge_index,
            reference_xyz,
            reference_time_steps,
            one_hot,
        ) = self.graph_information[idx]

        label = self.labels[idx]
        # TODO add node_features as data.pos
        data = pyg_data.Data(
            node_xyz=node_xyz,
            node_indices=node_indices,
            node_time_steps=node_time_steps,
            edge_index=edge_index,
            reference_xyz=reference_xyz,
            reference_time_steps=reference_time_steps,
            one_hot=one_hot,
            is_node=None,
            y=torch.tensor(label),
        )

        if self.transform:
            data = self.transform(data)

        return data
