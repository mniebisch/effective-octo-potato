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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference function of the classifier.

        The network uses a simple MLP per frame and than calcultes the mean of 
        the frames as its prediction. To confirm to the 
        torch.nn.CrossEntropyLoss, no softmax is calculated. The structure of
        the network is 
        First Layer --------------------------------- + -> Prediction
                    \                                /
                     -> Second Layer -> Third Layer -
        .
        """
        x = x.view(1, -1, self.nr_inputs).transpose(1, 2)


        x = self.conv1(x)
        y = self.relu(x)
        y = self.conv2(y)
        y = self.relu(y)
        y = self.conv3(y)
        x = x + y

        return torch.mean(x, dim=-1)

class SimplerNet(torch.nn.Module):
    """Predicts the labels for a representation of all frames."""

    def __init__(self, nr_inputs: int = 543 * 4, nr_outputs: int = 250):
        super().__init__()

        self.fc1 = torch.nn.Linear(nr_inputs, nr_outputs)
        self.fc2 = torch.nn.Linear(nr_outputs, 1000)
        self.fc3 = torch.nn.Linear(1000, nr_outputs)
        
        self.relu = torch.nn.ReLU()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Simple MLP:

        First Layer --------------------------------- + -> Prediction
                    \                                /
                     -> Second Layer -> Third Layer -
        .
        """
        x = self.fc1(x)
        x = self.relu(x)
        y = self.fc2(x)
        y = self.relu(y)
        y = self.fc3(y)
        return x + y

if __name__ == "__main__":
    simplenet = SimpleNet()
    x = torch.randn(1, 10 * 543, 3)

    result = simplenet(x)
    print(result.shape)
    print(result)
