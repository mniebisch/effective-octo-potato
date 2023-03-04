"""Submodule to handle input data."""

import pathlib

import pandas as pd
import torch

__all__ = [
    "load_parquet_file",
]


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
