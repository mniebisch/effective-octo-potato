"""Submodule to handle input data."""

import pathlib

import pandas as pd
import torch
from torch.utils import data as torch_data


__all__ = [
    "LandmarkDataset",
    "load_parquet_file",
]

class LandmarkDataset(torch_data.Dataset):
    def __init__(self, data_csv: pathlib.Path) -> None:
        self._data = pd.read_csv(data_csv)

    def __len__(self) -> int:
        return self._data.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        parquet_file = self._data[idx]['path']
        landmark_data: torch.Tensor = load_parquet_file(parquet_file)
        sign: str = self._data[idx]['sign']
        # TODO map sign str to int
        # TODO have fun with the data paths
        return landmark_data, 


def load_parquet_file(file_path: pathlib.Path) -> torch.Tensor:
    """
    Load landmarks from parquet file.

    Load x, y, z coordinates from parquet file.


    Return:
    ------
        Torch tensor with shape [n_frames * n_landmarks, 3].
    """
    coord_columns = ["x", "y", "z"]
    sign_data = pd.read_parquet(path=file_path)
    return torch.Tensor(sign_data[coord_columns].values)
