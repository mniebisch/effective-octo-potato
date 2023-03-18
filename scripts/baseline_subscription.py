import functools
import json
import multiprocessing as mp
import pathlib

import numpy as np
import pandas as pd
import torch
import tqdm
from numpy import typing as npt
from torch.utils import data as torch_data


def _create_features(
    file_name: pathlib.Path, feature_generator: torch.nn.Module
) -> npt.NDArray[np.float32]:
    sign_sequence = load_relevant_data_subset(pq_path=file_name)
    sign_sequence = torch.from_numpy(sign_sequence)
    features = feature_generator(sign_sequence)
    return features.cpu().numpy()


class FeatureGenerator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Feature Transformation.

        Args:
            x: Sign sequence with shape:
               [num frames, num landmarks, num spatial dims]
               [num frames, 543, 3] of datatype float32.


        Returns:
            Feature vector representing sign sequence with shape
            [num selected landmarks * num spatial coordinates  * num features].
            Eg. num landmarks == 85 if only left hand, right hand, and pose are
            used.

        """
        # input_shape = x.shape
        # if not len(input_shape) == 3 and not input_shape[2] == 3:
        #     raise ValueError("Invalid input shape.")
        # x = torch.from_numpy(x)
        x = self._select_landmarks(x)
        features = self._compute_features(x)
        return self._shape_features(features)

    def _compute_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean_features = torch.nanmean(x, dim=0)
        median_features = torch.nanmean(x, dim=0)
        return (torch.nan_to_num(mean_features), torch.nan_to_num(median_features))

    def _select_landmarks(self, x: torch.Tensor) -> torch.Tensor:
        left_hand = torch.arange(458, 489, dtype=torch.long)
        right_hand = torch.arange(522, 543, dtype=torch.long)
        pose = torch.arange(489, 522, dtype=torch.long)
        landmark_indices = torch.cat([left_hand, right_hand, pose])
        return x[:, landmark_indices, :]

    def _shape_features(
        self, features: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        features = torch.concatenate(features, dim=1)
        return torch.flatten(features)


class Dataset(torch_data.Dataset):
    def __init__(
        self, feature_matrix: npt.NDArray[np.float32], labels: npt.NDArray[np.integer]
    ) -> None:
        self.feature_matrix = feature_matrix
        self.labels = labels

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        features = self.feature_matrix[idx]
        return torch.Tensor(features), self.labels[idx]

    def __len__(self) -> int:
        return self.feature_matrix.shape[0]


class Net(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        width1 = 1024 * 10
        width2 = 512 * 10
        self.fc1 = torch.nn.Linear(input_dim, width1)
        self.fc2 = torch.nn.Linear(width1, width2)
        self.fc3 = torch.nn.Linear(width2, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def load_relevant_data_subset(pq_path: pathlib.Path) -> npt.NDArray[np.float32]:
    ROWS_PER_FRAME = 543  # number of landmarks per frame
    data_columns = ["x", "y", "z"]
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)


def create_features(
    file_names: list[pathlib.Path], feature_generator: torch.nn.Module
) -> npt.NDArray[np.float32]:
    _create_features_fg = functools.partial(
        _create_features, feature_generator=feature_generator
    )
    with mp.Pool(mp.cpu_count()) as pool:
        features = tqdm.tqdm(
            pool.imap(_create_features_fg, file_names),
            total=len(file_names),
        )
        features = list(features)
    return np.stack(features)


def eval(
    model: torch.nn.Module,
    dataloader: torch_data.DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        batch_iterator = tqdm.tqdm(dataloader)
        for batch_data, batch_label in batch_iterator:
            batch_data = batch_data.to(device)
            prediction = model(batch_data)
            prediction_label = torch.argmax(prediction, dim=1)
            prediction_label = prediction_label.detach().cpu()
            total += batch_label.shape[0]
            correct += (batch_label == prediction_label).sum().item()
    return correct / total


if __name__ == "__main__":
    data_base_path = pathlib.Path(__file__).parent.parent / "data"

    # hyperparamters
    batch_size = 64
    epochs = 1

    # load/process labels
    data_csv = "train.csv"
    train_df = pd.read_csv(data_base_path / data_csv)
    label_csv = "sign_to_prediction_index_map.json"
    labels = train_df["sign"]
    with open(data_base_path / label_csv) as file:
        label_map = json.load(file)
    labels = labels.replace(label_map)
    labels: npt.NDArray[np.integer] = labels.values

    example_file = data_base_path / train_df["path"][1]

    # create/load features
    feature_matrix_file_name = "baseline_mean_median2.npy"
    feature_matrix_file_name = data_base_path / feature_matrix_file_name
    fg = FeatureGenerator()
    if not feature_matrix_file_name.is_file():
        file_paths = [
            data_base_path / filename for filename in train_df["path"].tolist()
        ]
        feature_matrix = create_features(file_paths, fg)
        np.save(feature_matrix_file_name, feature_matrix)
    else:
        feature_matrix = np.load(feature_matrix_file_name)

    # split data
    split_ind = 70000
    train_matrix, valid_matrix = feature_matrix[:split_ind], feature_matrix[split_ind:]
    train_labels, valid_labels = labels[:split_ind], labels[split_ind:]

    train_dataset = Dataset(train_matrix, train_labels)
    train_dataloader = torch_data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_dim = train_matrix.shape[1]
    output_dim = len(label_map)
    model = Net(input_dim=input_dim, output_dim=output_dim)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=10e-2)

    model.train()
    for epoch_idx in range(epochs):
        batch_iterator = tqdm.tqdm(
            train_dataloader, desc=f"Epoch: {epoch_idx+1:02d}/{epochs}"
        )
        for batch_data, batch_label in batch_iterator:
            batch_data, batch_label = batch_data.to(device), batch_label.to(device)
            optimizer.zero_grad()
            prediction = model(batch_data)
            loss = criterion(prediction, batch_label)
            loss.backward()
            optimizer.step()
            batch_iterator.set_postfix({"loss": loss.item()})

    # evaluation
    valid_dataset = Dataset(valid_matrix, valid_labels)
    valid_dataloader = torch_data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False
    )
    train_acc = eval(model, train_dataloader, device)
    valid_acc = eval(model, valid_dataloader, device)
    print("train acc", train_acc)
    print("valid acc", valid_acc)

    # transform to onnx
    feature_sample_input = torch.rand((50, 543, 3))
    feature_onnx_file = "feature_generator.onnx"
    feature_onnx_file = data_base_path / feature_onnx_file
    fg.eval()
    torch.onnx.export(
        fg,
        feature_sample_input,
        feature_onnx_file,
        opset_version=12,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "input"}},
    )
