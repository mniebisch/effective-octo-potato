import torch

__all__ = ["apply_node_mask_to_edges"]


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
    valid_edges = mask[edge_index].all(dim=0)  # check if 1 is correct\
    edge_index = edge_index[:, valid_edges]

    # create lookup
    lookup_table = torch.zeros_like(mask, dtype=torch.long)
    lookup_table[mask] = torch.arange(torch.sum(mask).item())

    return lookup_table[edge_index]
