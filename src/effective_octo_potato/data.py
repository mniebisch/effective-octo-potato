"""
Submodule to handle input data.

author = Michael Niebisch, Christian Reimers
"""
import json
import pathlib

import numpy as np
import pandas as pd
import torch
from torch.utils import data as torch_data

__all__ = [
    "LandmarkDataset",
    "load_parquet_file",
]

class LandmarkDataset(torch_data.Dataset):
    """Provide the dataset for the ladmark data."""

    def __init__(
            self,
            data_dir: pathlib.Path,
            data_csv: pathlib.Path,
            label_csv: pathlib.Path,
    ):
        self._data = pd.read_csv(data_dir / data_csv)
        self.labels = json.loads(data_dir / label_csv.open().read())
        self.data_dir = data_dir

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return self._data.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Provide the example for a specific index."""
        parquet_file = self._data[idx]["path"]
        landmark_data: torch.Tensor = load_parquet_file(self.data_dir / parquet_file)
        sign: str = self._data[idx]["sign"]
        label = self.labels[sign]
        return landmark_data, label


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
    sign_data_numpy = np.array(sign_data[coord_columns].values)
    sign_data_numpy = np.nan_to_num(sign_data_numpy)
    return torch.Tensor(sign_data_numpy)

