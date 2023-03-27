import torch
import torch_geometric.utils as pyg_utils

__all__ = [
    "apply_node_mask_to_edges",
    "create_edge_index",
    "create_left_hand_edge_index",
    "create_node_indices",
    "create_node_mask",
    "create_pose_edge_index",
    "create_right_hand_edge_index",
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


def create_node_mask() -> torch.Tensor:
    num_input_nodes = 543
    node_indices = create_node_indices()
    node_mask = torch.zeros(num_input_nodes, dtype=torch.bool)
    node_mask[node_indices] = True
    return node_mask


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
