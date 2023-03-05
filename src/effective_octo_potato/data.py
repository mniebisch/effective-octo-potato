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
            *, 
            ignore_z: bool = False
    ):
        self._data = list(pd.read_csv(data_dir / data_csv)["path"])
        self._label = list(pd.read_csv(data_dir / data_csv)["sign"])
        self.labels = json.loads((data_dir / label_csv).open().read())
        self.data_dir = data_dir
        self.ignore_z = ignore_z

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self._data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Provide the example for a specific index."""
        parquet_file = self._data[idx]
        landmark_data: torch.Tensor = load_parquet_file(
                self.data_dir / parquet_file, 
                ignore_z = self.ignore_z,
            )
        sign: str = self._label[idx]
        label = self.labels[sign]
        return landmark_data, label


def load_parquet_file(
        file_path: pathlib.Path, 
        *, 
        ignore_z: bool = False,
) -> torch.Tensor:
    """
    Load landmarks from parquet file.

    Load x, y, z coordinates from parquet file.


    Return:
    ------
        Torch tensor with shape [n_frames * n_landmarks, 3].
    """
    coord_columns = ["x", "y", "z"]
    if ignore_z:
        coord_columns = ["x", "y"]
    sign_data = pd.read_parquet(path=file_path)
    sign_data_numpy = np.array(sign_data[coord_columns].values)
    sign_data_numpy = np.nan_to_num(sign_data_numpy)
    return torch.Tensor(sign_data_numpy)

