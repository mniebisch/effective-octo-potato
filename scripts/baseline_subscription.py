"""
Script based on:
https://www.kaggle.com/code/mayukh18/end-to-end-pytorch-training-submission

Usefule links:
    - https://glaringlee.github.io/onnx.html#supported-operators

"""

import functools
import json
import multiprocessing as mp
import pathlib
from typing import Dict, List, Tuple

import numpy as np
import onnx
import pandas as pd
import tensorflow as tf
import torch
import tqdm
from numpy import typing as npt
from onnx_tf import backend as onnx_tf_backend
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
        lefth_x = x[:, 468:489, :].contiguous().view(-1, 21 * 3)
        pose_x = x[:, 489:522, :].contiguous().view(-1, 33 * 3)
        righth_x = x[:, 522:, :].contiguous().view(-1, 21 * 3)

        lefth_x = lefth_x[~torch.any(torch.isnan(lefth_x), dim=1), :]
        righth_x = righth_x[~torch.any(torch.isnan(righth_x), dim=1), :]

        x2m = torch.mean(lefth_x, 0)
        x3m = torch.mean(pose_x, 0)
        x4m = torch.mean(righth_x, 0)

        x2s = torch.std(lefth_x, 0)
        x3s = torch.std(pose_x, 0)
        x4s = torch.std(righth_x, 0)

        xfeat = torch.cat([x2m, x3m, x4m, x2s, x3s, x4s], axis=0)
        xfeat = torch.where(
            torch.isnan(xfeat), torch.tensor(0.0, dtype=torch.float32), xfeat
        )

        return xfeat


class Dataset(torch_data.Dataset):
    def __init__(
        self, feature_matrix: npt.NDArray[np.float32], labels: npt.NDArray[np.integer]
    ) -> None:
        self.feature_matrix = feature_matrix
        self.labels = labels

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
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
    file_names: List[pathlib.Path], feature_generator: torch.nn.Module
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


def _export_torch_to_onnx(
    model: torch.nn.Module, sample: torch.Tensor, output_file: pathlib.Path
) -> None:
    model.eval()
    torch.onnx.export(
        model,
        sample,
        output_file,
        opset_version=12,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "input"}},
    )


def _export_onnx_to_tensorflow(
    onnx_file: pathlib.Path, tensorflow_graph_path: pathlib.Path
) -> None:
    onnx_representation = onnx.load(onnx_file)
    tensorflow_representation = onnx_tf_backend.prepare(onnx_representation)
    tensorflow_representation.export_graph(tensorflow_graph_path)


def _create_tensorflow_inference_model(
    feature_generator_path: pathlib.Path,
    model_path: pathlib.Path,
    output_path: pathlib.Path,
) -> None:
    class NetInference(tf.Module):
        def __init__(self):
            super().__init__()
            self.feature_generator = tf.saved_model.load(feature_generator_path)
            self.model = tf.saved_model.load(model_path)
            self.feature_generator.trainable = False
            self.model.trainable = False

        @tf.function(
            input_signature=[
                tf.TensorSpec(
                    shape=[None, 543, 3],
                    dtype=tf.float32,
                    name="inputs",
                )
            ]
        )
        def call(self, input):
            output_tensors = {}
            features = self.feature_generator(**{"input": input})["output"]
            output_tensors["outputs"] = self.model(
                **{"input": tf.expand_dims(features, 0)}
            )["output"][0, :]
            return output_tensors

    tensorflow_model = NetInference()
    output_path: str = str(output_path)
    tf.saved_model.save(
        tensorflow_model,
        output_path,
        signatures={"serving_default": tensorflow_model.call},
    )


def _export_tensorflow_lite_model(
    inference_model_path: pathlib.Path, tensorflow_lite_file: pathlib.Path
) -> None:
    inference_model_path: str = str(inference_model_path)
    converter = tf.lite.TFLiteConverter.from_saved_model(inference_model_path)
    tflite_model = converter.convert()

    with open(tensorflow_lite_file, "wb") as f:
        f.write(tflite_model)


def transform_model(
    model: torch.nn.Module,
    feature_generator: torch.nn.Module,
    data_path: pathlib.Path,
    num_model_input_features: int,
) -> None:
    # transform feature generator to onnx
    feature_sample_input = torch.rand((50, 543, 3))
    feature_onnx_file = "feature_generator.onnx"
    feature_onnx_file = data_path / feature_onnx_file
    _export_torch_to_onnx(
        model=feature_generator,
        sample=feature_sample_input,
        output_file=feature_onnx_file,
    )

    # transform model to onnx
    model_sample_input = torch.rand((1, num_model_input_features)).cuda()
    model_onnx_file = "model.onnx"
    model_onnx_file = data_path / model_onnx_file
    _export_torch_to_onnx(
        model=model, sample=model_sample_input, output_file=model_onnx_file
    )

    # transform feature generator to tensorflow
    feature_tensorflow_path = data_path / "tf_feature_generator"
    _export_onnx_to_tensorflow(
        onnx_file=feature_onnx_file, tensorflow_graph_path=feature_tensorflow_path
    )

    # transform model to tensorflow
    model_tensorflow_path = data_path / "tf_model"
    _export_onnx_to_tensorflow(
        onnx_file=model_onnx_file, tensorflow_graph_path=model_tensorflow_path
    )

    # create tensorflow inference model
    inference_model_tensorflow_path: pathlib.Path = data_path / "inference_model"
    _create_tensorflow_inference_model(
        feature_generator_path=feature_tensorflow_path,
        model_path=model_tensorflow_path,
        output_path=inference_model_tensorflow_path,
    )

    # create tensorflow lite model for submission
    tflite_model_file = "model.tflite"
    tflite_model_file = data_path / tflite_model_file

    _export_tensorflow_lite_model(
        inference_model_path=inference_model_tensorflow_path,
        tensorflow_lite_file=tflite_model_file,
    )


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


def _get_label_map(data_dir: pathlib.Path) -> Dict[str, int]:
    label_csv = "sign_to_prediction_index_map.json"
    with open(data_dir / label_csv) as file:
        label_map = json.load(file)
    return label_map


def load_labels(data_dir: pathlib.Path, labels: pd.Series) -> npt.NDArray[np.integer]:
    label_map = _get_label_map(data_dir=data_dir)
    labels = labels.replace(label_map)
    return labels.values


def handle_training_data(
    raw_dir: pathlib.Path,
    feature_dir: pathlib.Path,
    feature_generator: torch.nn.Module,
    feature_file_name: str,
) -> npt.NDArray[np.float32]:
    feature_matrix_file_name = feature_dir / feature_file_name
    if not feature_matrix_file_name.is_file():
        file_paths = [raw_dir / filename for filename in train_df["path"].tolist()]
        feature_matrix = create_features(file_paths, feature_generator)
        np.save(feature_matrix_file_name, feature_matrix)
    else:
        feature_matrix = np.load(feature_matrix_file_name)
    return feature_matrix


if __name__ == "__main__":
    data_base_path = pathlib.Path(__file__).parent.parent / "data"
    output_base_path = pathlib.Path(__file__).parent.parent / "data"

    # hyperparamters
    batch_size = 64
    epochs = 1

    # load/process labels
    data_csv = "train.csv"
    train_df = pd.read_csv(data_base_path / data_csv)
    labels = load_labels(data_dir=data_base_path, labels=train_df["sign"])

    # create/load features
    feature_matrix_file_name = "baseline_mean_std_kaggle.npy"
    fg = FeatureGenerator()
    feature_matrix = handle_training_data(
        raw_dir=data_base_path,
        feature_dir=output_base_path,
        feature_generator=fg,
        feature_file_name=feature_matrix_file_name,
    )

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
    output_dim = len(_get_label_map(data_base_path))
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

    transform_model(
        model=model,
        feature_generator=fg,
        data_path=output_base_path,
        num_model_input_features=450,
    )
