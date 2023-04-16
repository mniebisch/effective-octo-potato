import torch
from torch_geometric import data as pyg_data
from torch_geometric import transforms as pyg_transforms
from torch_geometric.data import datapipes as pyg_datapipes

from effective_octo_potato.graph_utils import calc_node_dist_to_reference_feature

__all__ = [
    "CalcReferenceFeatures",
    "CatNodeFeatures",
    "PosSplitNodes",
    "PosStackNodes",
]


@pyg_datapipes.functional_transform("calc_reference_features")
class CalcReferenceFeatures(pyg_transforms.BaseTransform):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, data: pyg_data.Data) -> pyg_data.Data:
        reference_features = []
        for time_ind in torch.unique(data.node_time_steps):
            reference_feature_time_step = calc_node_dist_to_reference_feature(
                nodes=data.node_xyz[data.node_time_steps == time_ind],
                reference=data.reference_xyz[data.reference_time_steps == time_ind],
            )
            reference_features.append(reference_feature_time_step)

        data.reference_features = torch.cat(reference_features, dim=0)
        return data


@pyg_datapipes.functional_transform("cat_node_features")
class CatNodeFeatures(pyg_transforms.BaseTransform):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, data: pyg_data.Data) -> pyg_data.Data:
        data.x = torch.cat(
            [data.node_xyz, data.reference_features, data.one_hot], dim=1
        )
        return data


# hmpf introduces coupling
@pyg_datapipes.functional_transform("pos_stack_nodes")
class PosStackNodes(pyg_transforms.BaseTransform):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, data: pyg_data.Data) -> pyg_data.Data:
        data.pos = torch.cat([data.node_xyz, data.reference_xyz], dim=0)
        is_node = torch.zeros(data.pos.shape[0], dtype=torch.bool)
        is_node[torch.arange(data.node_xyz.shape[0])] = True
        data.is_node = is_node
        return data


# hmpf introduces coupling
@pyg_datapipes.functional_transform("pos_split_nodes")
class PosSplitNodes(pyg_transforms.BaseTransform):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, data: pyg_data.Data) -> pyg_data.Data:
        data.node_xyz = data.pos[data.is_node]
        data.reference_xyz = data.pos[torch.logical_not(data.is_node)]
        data.is_node = None
        data.pos = None
        return data
