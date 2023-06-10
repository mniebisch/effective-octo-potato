import torch
import torch_geometric.nn as pyg_nn

__all__ = ["GCN"]


class MeinBlock(torch.nn.Module):
    def __init__(
        self, input_channels: int, hidden_channels: int, aggr: str = "sum"
    ) -> None:
        super().__init__()
        self.conv1 = pyg_nn.GATv2Conv(
            input_channels, hidden_channels, edge_dim=2, aggr=aggr
        )
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.act1 = torch.nn.PReLU()
        self.conv2 = pyg_nn.GATv2Conv(
            input_channels, hidden_channels, edge_dim=2, aggr=aggr
        )
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.act2 = torch.nn.PReLU()

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        residual = x
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = residual + x
        x = self.act2(x)
        return x


class GCN(torch.nn.Module):
    def __init__(
        self,
        num_node_features: int,
        hidden_channels: int,
        num_classes: int,
        num_blocks: int,
        drop_p: float,
    ) -> None:
        super().__init__()

        self.pre_fc1 = torch.nn.Linear(num_node_features, hidden_channels)
        self.pre_bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.pre_act1 = torch.nn.PReLU()
        self.blocks = torch.nn.ModuleList(
            [MeinBlock(hidden_channels, hidden_channels) for _ in range(num_blocks)]
        )
        self.post_fc1 = torch.nn.Linear(hidden_channels, hidden_channels * 4)
        self.post_bn1 = torch.nn.BatchNorm1d(hidden_channels * 4)
        self.post_act1 = torch.nn.PReLU()
        self.post_fc2 = torch.nn.Linear(hidden_channels * 4, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        x = self.pre_fc1(x)
        x = self.pre_bn1(x)
        x = self.pre_act1(x)
        for i in range(len(self.blocks)):
            x = self.blocks[i](x, edge_index, edge_attr)
        x = pyg_nn.global_max_pool(x, batch)
        x = self.post_fc1(x)
        x = self.post_bn1(x)
        x = self.post_act1(x)
        x = self.post_fc2(x)
        return x
