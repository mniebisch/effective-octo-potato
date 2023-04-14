import torch
from torch_geometric import data as pyg_data
from torch_geometric import transforms as pyg_transforms
from torch_geometric.data import datapipes as pyg_datapipes

from effective_octo_potato.graph_utils import calc_node_dist_to_reference_feature

__all__ = ["CalcReferenceFeatures"]


@pyg_datapipes.functional_transform("calc_reference_features")
class CalcReferenceFeatures(pyg_transforms.BaseTransform):
    def __init__(self) -> None:
        pass

    def __call__(self, data: pyg_data.Data) -> pyg_data.Data:
        nodes = data.pos[data.node_indices]
        node_time_steps = data.time_steps[data.node_indices]
        reference = data.pos[data.reference_indices]
        reference_time_steps = data.time_steps[data.reference_indices]

        reference_features = []
        for time_ind in torch.unique(data.time_steps):
            reference_feature_time_step = calc_node_dist_to_reference_feature(
                nodes=nodes[node_time_steps == time_ind],
                reference=reference[reference_time_steps == time_ind],
            )
            reference_features.append(reference_feature_time_step)
        reference_features = torch.cat(reference_features)

        # TODO init x? empty but correct shape? something like torch.tensor([[] ])
        data.x = torch.cat([data.x, nodes, reference_features], dim=1)

        return data


# prep for x init
# torch.tensor([[] for _ in range(len(data.node_indices))])
