import torch
import torch_geometric.utils as pyg_utils

__all__ = [
    "apply_node_mask_to_edges",
    "create_edge_index",
    "create_left_hand_edge_index",
    "create_node_indices",
    "create_node_mask",
    "create_one_hot",
    "create_pose_edge_index",
    "create_right_hand_edge_index",
    "get_pose_subgraph_nodes",
]


def apply_node_mask_to_edges(
    mask: torch.Tensor, edge_index: torch.Tensor
) -> torch.Tensor:
    """_summary_

    Mapping filtered nodes (changed indices) to edge_indices.

    Args:
        mask (torch.Tensor): Of type boolean with shape [num_nodes, *].
            True indicates value is kept.
        edge_index (torch.Tensor): Shape [2, num_edges]

    Returns:
        torch.Tensor: _description_
    """
    # drop edges, keep where col(!) is all true
    valid_edges = mask[edge_index].all(dim=0)
    edge_index = edge_index[:, valid_edges]

    # create lookup
    lookup_table = torch.zeros_like(mask, dtype=torch.long)
    lookup_table[mask] = torch.arange(torch.sum(mask).item())

    return lookup_table[edge_index]


def create_node_indices() -> torch.Tensor:
    """
    The node indices in our experiments.
    We use the following body parts:
        - left hand
        - right hand
        - pose

    The numeration of the nodes is such that it matches the order or the full
    dataset (all body parts) provided at inference time (543 nodes per frame).
    """
    left_hand_indices = torch.arange(468, 489)
    pose_indices = torch.arange(489, 522)
    right_hand_indices = torch.arange(522, 543)
    return torch.cat([left_hand_indices, pose_indices, right_hand_indices])


def create_node_mask(
    node_indices: torch.Tensor, num_input_nodes: int = 543
) -> torch.Tensor:
    node_mask = torch.zeros(num_input_nodes, dtype=torch.bool)
    node_mask[node_indices] = True
    return node_mask


def create_one_hot(num_nodes: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(torch.arange(num_nodes))


def _create_hand_edge_index() -> torch.Tensor:
    hand_edges = torch.tensor(
        [
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
        ],
        dtype=torch.int64,
    )
    hand_edges = hand_edges.T
    return pyg_utils.to_undirected(hand_edges)


def create_left_hand_edge_index() -> torch.Tensor:
    left_hand_shift = 468
    return _create_hand_edge_index() + left_hand_shift


def create_right_hand_edge_index() -> torch.Tensor:
    right_hand_shift = 522
    return _create_hand_edge_index() + right_hand_shift


def _create_pose_edge_index() -> torch.Tensor:
    pose_edges = torch.tensor(
        [
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
        ],
        dtype=torch.int64,
    )
    pose_edges = pose_edges.T
    return pyg_utils.to_undirected(pose_edges)


def create_pose_edge_index() -> torch.Tensor:
    pose_shift = 489
    return _create_pose_edge_index() + pose_shift


def create_edge_index() -> torch.Tensor:
    """
    The edge indices for the nodes we are using in our experiments.
    We use the following body parts:
        - left hand
        - right hand
        - pose

    The numeration of the nodes is such that it matches the order or the full
    dataset (all body parts) provided at inference time (543 nodes per frame).
    """
    body_part_edge_index = [
        create_left_hand_edge_index(),
        create_right_hand_edge_index(),
        create_pose_edge_index(),
    ]
    return torch.cat(body_part_edge_index, dim=1)


def get_pose_subgraph_nodes() -> torch.Tensor:
    # relative indices (only considering indices of pose)
    relative_pose_indices = torch.arange(11, 23)
    # absolute indices (considering nodes from all body parts)
    absolute_pose_indices = relative_pose_indices + 489
    return absolute_pose_indices


def compute_reference_nodes(node_coords: torch.Tensor) -> torch.Tensor:
    # node_coords = torch.zeros((543, 3))
    plain_reference_ind = torch.tensor(
        [
            8,  # right ear
            7,  # left ear
            5,  # right eye
            2,  # left eye
            0,  # nose
            10,  # mouth right
            9,  # mouth left
            12,  # right shoulder
            11,  # left shoulder
            24,  # right hip
            23,  # left hip
        ],
        dtype=torch.long,
    )

    right_hand_reference_ind = torch.tensor(
        [20, 22, 18, 16], dtype=torch.long
    )  # right hand
    left_hand_reference_ind = torch.tensor(
        [19, 21, 15, 17], dtype=torch.long
    )  # left hand

    agg_reference_right_hand_nodes = torch.mean(
        node_coords[right_hand_reference_ind], dim=0, keepdim=True
    )
    agg_reference_left_hand_nodes = torch.mean(
        node_coords[left_hand_reference_ind], dim=0, keepdim=True
    )
    plain_reference_nodes = node_coords[plain_reference_ind]

    return torch.cat(
        [
            plain_reference_nodes,
            agg_reference_right_hand_nodes,
            agg_reference_left_hand_nodes,
        ]
    )


def calc_node_dist_to_reference_feature(
    nodes: torch.Tensor, reference: torch.Tensor
) -> torch.Tensor:
    """
    For each node in nodes compute distance to each node in reference.

    Args:
        nodes:
            Spatial coordinates of each nodes.
            Input shape [num_nodes, num_spatial_dims].
            These are the nodes for which we would like to compute the reference
            distance features.
        reference:
            Spatial coordinated of reference nodes.
            Input shape [num_reference_nodes, num_spatial_dims]
    Returns:
        Distance matrix with shape [num_nodes, num_reference_nodes].
        The output can be concatenate to nodes input via torch.cat in dim 1.
    """
    return torch.cdist(nodes, reference, p=2)
