import json
import multiprocessing as mp
import pathlib
import warnings

import numpy as np
import pandas as pd
import torch
import tqdm
from numpy import typing as npt
from torch.utils import data as torch_data


def create_features(file_names: list[pathlib.Path]) -> npt.NDArray[np.float32]:
    with mp.Pool(mp.cpu_count()) as pool:
        features = tqdm.tqdm(
            pool.imap(_create_features, file_names), total=len(file_names)
        )
        features = list(features)
    return np.stack(features)


def _select_data(sign_data: pd.DataFrame) -> npt.NDArray[np.floating]:
    # handle indices START TODO find better solution
    left_hand = list(range(468, 489))
    right_hand = list(range(522, 543))
    pose = list(range(489, 522))
    lip = [
        61,
        185,
        40,
        39,
        37,
        0,
        267,
        269,
        270,
        409,
        291,
        146,
        91,
        181,
        84,
        17,
        314,
        405,
        321,
        375,
        78,
        191,
        80,
        81,
        82,
        13,
        312,
        311,
        310,
        415,
        95,
        88,
        178,
        87,
        14,
        317,
        402,
        318,
        324,
        308,
    ]
    node_indices = left_hand + right_hand + pose
    # node_indices = list(range(543))
    # handle indices END
    required_columns = pd.Series(["x", "y", "z"])
    if not required_columns.isin(sign_data.columns).all():
        raise ValueError("Missing columns.")
    num_total_nodes = 543
    sign_coords = sign_data[["x", "y", "z"]].values
    sign_coords = np.reshape(sign_coords, (-1, num_total_nodes, 3))
    return sign_coords[:, node_indices, :]


def _compute_features(
    sign_coords: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    input_shape = sign_coords.shape
    if not len(input_shape) == 3 and not input_shape[2] == 3:
        raise ValueError
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, message=r"Mean of empty slice"
        )
        mean_feature = np.nanmean(sign_coords, axis=0)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, message=r"All-NaN slice encountered"
        )
        std_feature = np.nanmedian(sign_coords, axis=0)
    return np.nan_to_num(mean_feature), np.nan_to_num(std_feature)


def _shape_features(
    feature_data: tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]
) -> npt.NDArray[np.float32]:
    features = np.concatenate(feature_data, axis=1)
    return features.flatten()


def _load_sign_sequence(file_name: pathlib.Path) -> pd.DataFrame:
    return pd.read_parquet(file_name, columns=["x", "y", "z"])


def _create_features(file_name: pathlib.Path) -> npt.NDArray[np.float32]:
    sign_sequence = _load_sign_sequence(file_name)
    sign_sequence: npt.NDArray[np.floating] = _select_data(sign_sequence)
    sign_sequence: npt.NDArray[np.float32] = sign_sequence.astype(np.float32)
    features = _compute_features(sign_sequence)
    return _shape_features(features)


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


def eval(model: torch.nn.Module, dataloader: torch_data.DataLoader) -> float:
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
    epochs = 2

    # load/process labels
    data_csv = "train.csv"
    train_df = pd.read_csv(data_base_path / data_csv)
    label_csv = "sign_to_prediction_index_map.json"
    labels = train_df["sign"]
    with open(data_base_path / label_csv) as file:
        label_map = json.load(file)
    labels = labels.replace(label_map)
    labels: npt.NDArray[np.integer] = labels.values

    # create/load features
    feature_matrix_file_name = "baseline_mean_median.npy"
    feature_matrix_file_name = data_base_path / feature_matrix_file_name
    if not feature_matrix_file_name.is_file():
        file_paths = [
            data_base_path / filename for filename in train_df["path"].tolist()
        ]
        feature_matrix = create_features(file_paths)
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
    train_acc = eval(model, train_dataloader)
    valid_acc = eval(model, valid_dataloader)
    print("train acc", train_acc)
    print("valid acc", valid_acc)
