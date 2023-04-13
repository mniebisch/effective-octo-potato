import pathlib

import numpy as np
import numpy.typing as npt
import pandas as pd

__all__ = ["load_relevant_data_subset"]


def load_relevant_data_subset(pq_path: pathlib.Path) -> npt.NDArray[np.float32]:
    ROWS_PER_FRAME = 543  # number of landmarks per frame
    data_columns = ["x", "y", "z"]
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)
