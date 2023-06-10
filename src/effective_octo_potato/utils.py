import json
import pathlib
from typing import Dict, List

import numpy as np
import numpy.typing as npt
import pandas as pd

__all__ = ["load_labels", "get_label_names"]


def load_labels(data_dir: pathlib.Path, labels: pd.Series) -> npt.NDArray[np.integer]:
    label_map = _get_label_map(data_dir=data_dir)
    labels = labels.replace(label_map)
    return labels.values


def _get_label_map(data_dir: pathlib.Path) -> Dict[str, int]:
    label_csv = "sign_to_prediction_index_map.json"
    with open(data_dir / label_csv) as file:
        label_map = json.load(file)
    return label_map


def get_label_names(data_dir: pathlib.Path) -> List[str]:
    return list(_get_label_map(data_dir).keys())
