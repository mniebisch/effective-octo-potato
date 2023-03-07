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


# TODO write test
def create_idx_map(data: pd.DataFrame) -> list[tuple[int, int]]:
    """Batch index map creation.

    Create a map to access data of a specific frame from data table.
    """
    # handle mapping to single frame
    frame_ranges: list[tuple[int, int]] = list(
        zip(data["frame_min"], data["frame_max"])
    )
    # # frame_ids with length (num frames per file * num files)
    frame_ids = [
        frame
        for frame_min, frame_max in frame_ranges
        for frame in range(frame_min, frame_max + 1)
    ]
    frames_num = [max_frame - min_frame + 1 for min_frame, max_frame in frame_ranges]
    file_idxs = np.repeat(range(data.shape[0]), frames_num).tolist()
    assert len(frame_ids) == len(file_idxs)  # remove assert after writing tests.
    return list(zip(file_idxs, frame_ids))


class LandmarkDataset(torch_data.Dataset):
    """Provide the dataset for the ladmark data."""

    def __init__(
        self,
        data_dir: pathlib.Path,
        data_csv: pathlib.Path,
        label_csv: pathlib.Path,
        *,
        ignore_z: bool = False,
    ):
        required_columns = pd.Series(["path", "sign", "frame_min", "frame_max"])
        data = pd.read_csv(data_dir / data_csv)
        if not required_columns.isin(data.columns).all():
            raise ValueError(
                f"Input dataframe (data_csv) is required to have columns {data}."
            )
        self._data = list(data["path"])
        self._label = list(data["sign"])
        self.labels = json.loads((data_dir / label_csv).open().read())
        self.data_dir = data_dir
        self.ignore_z = ignore_z

        self._idx_map: list[tuple[int, int]] = create_idx_map(data)

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self._idx_map)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Provide the example for a specific index."""
        parquet_file_idx, frame_idx = self._idx_map[idx]
        parquet_file = self._data[parquet_file_idx]
        landmark_data: torch.Tensor = load_parquet_file(
            self.data_dir / parquet_file,
            ignore_z=self.ignore_z,
        )
        landmark_data = landmark_data[landmark_data["frame"] == frame_idx]
        sign: str = self._label[parquet_file_idx]
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
