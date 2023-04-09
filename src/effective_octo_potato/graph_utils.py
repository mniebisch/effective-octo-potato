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
    "create_temporal_edge_indices",
    "fix_num_frames",
    "get_pose_subgraph_nodes",
    "map_edge_indices_to_temporal_graph",
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
    left_hand_indices = get_left_hand_nodes()
    pose_indices = get_pose_nodes()
    right_hand_indices = get_right_hand_nodes()
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


def get_left_hand_nodes() -> torch.Tensor:
    return torch.arange(468, 489)


def get_right_hand_nodes() -> torch.Tensor:
    return torch.arange(522, 543)


def get_pose_nodes() -> torch.Tensor:
    return torch.arange(489, 522)


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

    plain_reference_ind = plain_reference_ind + 489
    right_hand_reference_ind = right_hand_reference_ind + 489
    left_hand_reference_ind = left_hand_reference_ind + 489

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


def map_edge_indices_to_temporal_graph(
    edge_index: torch.Tensor, num_nodes: int, num_frames: int
) -> torch.Tensor:
    """
    Create edge connections for each graph in temporal graph.
    Aim of this function is to create pyg compatible edge indices such that the graph
    of a given time step is connected.
    No temporal connection is created with this function.

    Args:
        edge_index: COO adjacency representation with shape [2, num_edges].
        num_nodes: Number of nodes per frame.
        num_frames: Number of timesteps in temporal graph.
    Returns:
        Adapted COO adjecency representation for each frame such that after
        following processing steps a all edge_indices can be joined into single
        representation.
        Shape [num_frames, 2, num_edges].
    """
    node_shift = torch.arange(num_frames) * num_nodes
    node_shift = node_shift.reshape((num_frames, 1, 1))

    frame_edge_indices = edge_index.unsqueeze(0).repeat(num_frames, 1, 1)

    return frame_edge_indices + node_shift


def create_temporal_edge_indices(num_nodes: int, num_frames: int) -> torch.Tensor:
    """
    Create temporal edges to create temporal graph.

    Returns:
        Temporal edge indices with shape [2, num_edges].
        num_edges == num_nodes * num_frames.
    """
    nodes = torch.arange(num_nodes, dtype=torch.float32).reshape(1, -1)
    start_frame = torch.arange(num_frames).reshape(-1, 1) * num_nodes
    dest_frame = torch.arange(1, num_frames + 1).reshape(-1, 1) * num_nodes

    from_node = start_frame + nodes
    to_node = dest_frame + nodes

    from_node = torch.flatten(from_node)
    to_node = torch.flatten(to_node)

    return torch.stack([from_node, to_node])


def fix_num_frames(x: torch.Tensor, num_sampling_time_steps: int) -> torch.Tensor:
    """

    Args:
        x: Nodes in time (num_frames) and space (x, y, z). Shape (num_frames, 543, 3).
        num_sampling_time_steps:
            The number of time steps to which the number of frames is fit.

    Return:
        Down/Up sampled temporal point cloud with shape
        [num_sampling_time_steps, 543, 3].

    """
    return torch.nn.functional.interpolate(x, num_sampling_time_steps, mode="linear")
