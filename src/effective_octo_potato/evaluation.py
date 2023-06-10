import pathlib
from typing import List

import numpy as np
import pandas as pd
import tqdm

__all__ = ["compute_available_rate", "compute_num_frames", "compute_stats"]


def compute_available_rate(df: pd.DataFrame) -> pd.Series:
    """
    Time series (frame) Availability rate of each body part.
    1.0 -> for every frame body part data is not NaN
    0.0 -> for every frame body part data is Nan
    """
    xyz_nan = pd.isna(df[["x", "y", "z"]])
    rows_wise_all_nan = xyz_nan.eq(xyz_nan.iloc[:, 0], axis=0).all(1)
    if not rows_wise_all_nan.all():
        raise ValueError

    def all_nan(x: pd.Series) -> bool:
        return x.isna().values.all()

    def any_nan(x: pd.Series) -> bool:
        return x.isna().values.any()

    bodyparts_framewise = df.groupby(["frame", "type"]).agg({"x": [all_nan, any_nan]})
    bodyparts_framewise.columns = bodyparts_framewise.columns.droplevel(0)
    # check if all body part nodes are either nan or valid float for each frame
    if not (bodyparts_framewise["all_nan"] == bodyparts_framewise["any_nan"]).all():
        raise ValueError

    bodyparts_framewise = bodyparts_framewise.reset_index()

    def part_available(x: pd.Series) -> float:
        return np.sum(np.logical_not(x.values)) / len(x)

    available_rate = bodyparts_framewise.groupby(["type"]).agg(
        {"all_nan": part_available}
    )
    # blub = blub.reset_index()
    available_rate = available_rate.rename(columns={"all_nan": "available_rate"})
    return available_rate["available_rate"]


def compute_num_frames(df: pd.DataFrame) -> int:
    return df["frame"].nunique()


def compute_stats(paths: List[pathlib.Path]) -> pd.DataFrame:
    rate_stat = []
    frame_num_stat = []
    for file_path in tqdm.tqdm(paths):
        sign_data = pd.read_parquet(file_path, columns=["frame", "type", "x", "y", "z"])
        frame_num_stat.append(compute_num_frames(sign_data))
        rate_data = compute_available_rate(sign_data)
        rate_data.index.name = None
        rate_stat.append(rate_data)

    stats = pd.DataFrame(rate_stat)
    stats["frame_num"] = frame_num_stat

    return stats.reset_index(drop=True)
