import torch
from torch_geometric import data as pyg_data
from torch_geometric import transforms as pyg_transforms
from torch_geometric.data import datapipes as pyg_datapipes

from effective_octo_potato.graph_utils import calc_node_dist_to_reference_feature

__all__ = ["CalcReferenceFeatures"]


@pyg_datapipes.functional_transform("calc_reference_features")
class CalcReferenceFeatures(pyg_transforms.BaseTransform):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, data: pyg_data.Data) -> pyg_data.Data:
        nodes = data.pos[data.is_node]
        node_time_steps = data.time_steps[data.is_node]
        reference = data.pos[torch.logical_not(data.is_node)]
        reference_time_steps = data.time_steps[torch.logical_not(data.is_node)]

        reference_features = []
        for time_ind in torch.unique(data.time_steps):
            reference_feature_time_step = calc_node_dist_to_reference_feature(
                nodes=nodes[node_time_steps == time_ind],
                reference=reference[reference_time_steps == time_ind],
            )
            reference_features.append(reference_feature_time_step)

        data.x = reference_features = torch.cat(reference_features)
        return data


@pyg_datapipes.functional_transform("cat_node_features")
class CatNodeFeatures(pyg_transforms.BaseTransform):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, data: pyg_data.Data) -> pyg_data.Data:
        data.x = torch.cat([data.pos, data.reference_features, data.one_hot], dim=1)
        return data


# hmpf introduces coupling
class StackPosNodes(pyg_transforms.BaseTransform):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, data: pyg_data.Data) -> pyg_data.Data:
        data.pos = torch.cat([data.node_features, data.reference_features])
        is_node = torch.zeros_like(data.pos, dtype=torch.bool)
        is_node[torch.arange(data.num_node_features.shape[0])] = True
        data.is_node = is_node
        return data


# hmpf introduces coupling
class SplitPosNodes(pyg_transforms.BaseTransform):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, data: pyg_data.Data) -> pyg_data.Data:
        data.node_features = data.pos[data.is_node]
        data.reference_features = data.pos[torch.logical_not(data.is_node)]
        data.is_node = None
        return data
