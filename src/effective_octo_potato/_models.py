"""
File that contais the pdtorch models.

author = Christian Reimers
"""
import torch


class SimpleNet(torch.nn.Module):
    """Predicts the labels frame-wise and averages the predictions."""

    def __init__(self, nr_inputs: int = 543 * 3, nr_outputs: int = 250):
        super().__init__()

        self.nr_inputs = nr_inputs
        self.conv1 = torch.nn.Conv1d(
            in_channels=nr_inputs,
            out_channels=nr_outputs,
            kernel_size=1,
        )
        self.conv2 = torch.nn.Conv1d(
            in_channels=nr_outputs,
            out_channels=1000,
            kernel_size=1,
        )
        self.conv3 = torch.nn.Conv1d(
            in_channels=1000,
            out_channels=nr_outputs,
            kernel_size=1,
        )

        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inference function of the classifier."""
        x = x.view(1, -1, self.nr_inputs).transpose(1, 2)


        x = self.conv1(x)
        y = self.relu(x)
        y = self.conv2(y)
        y = self.relu(y)
        y = self.conv3(y)
        x = x + y
        x = self.softmax(x)

        return torch.mean(x, dim=-1)


if __name__ == "__main__":
    simplenet = SimpleNet()
    x = torch.randn(1, 10 * 543, 3)

    result = simplenet(x)
    print(result.shape)
    print(result)
