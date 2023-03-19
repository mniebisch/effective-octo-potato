"""
Copy of a kaggle notebook.

https://www.kaggle.com/code/mayukh18/end-to-end-pytorch-training-submission.

author = Christian Reimers
"""
from __future__ import annotations

import gc
import json
import pathlib
from typing import TYPE_CHECKING

import numpy as np
import onnx
import onnx_tf
import pandas as pd
import pathos.multiprocessing as mp
import tensorflow as tf
import torch
import tqdm

if TYPE_CHECKING:
    import numpy.typing as npt


class FeatureGen(torch.nn.Module):
    """Convert the (n_frames, 543,3) data to (n_features,) form."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert the features."""
        face_x = x[:, :468, :].contiguous().view(-1, 468 * 3)
        lefth_x = x[:, 468:489, :].contiguous().view(-1, 21 * 3)
        pose_x = x[:, 489:522, :].contiguous().view(-1, 33 * 3)
        righth_x = x[:, 522:, :].contiguous().view(-1, 21 * 3)

        lefth_x = lefth_x[~torch.any(torch.isnan(lefth_x), dim=1), :]
        righth_x = righth_x[~torch.any(torch.isnan(righth_x), dim=1), :]

        x1m = torch.mean(face_x, 0)
        x2m = torch.mean(lefth_x, 0)
        x3m = torch.mean(pose_x, 0)
        x4m = torch.mean(righth_x, 0)

        x1s = torch.std(face_x, 0)
        x2s = torch.std(lefth_x, 0)
        x3s = torch.std(pose_x, 0)
        x4s = torch.std(righth_x, 0)

        xfeat = torch.cat([x1m, x2m, x3m, x4m, x1s, x2s, x3s, x4s], axis=0)
        return torch.where(
            torch.isnan(xfeat),
            torch.tensor(0.0, dtype=torch.float32),
            xfeat,
        )


class ASLModel(torch.nn.Module):
    """
    The actual model.

    :param p: The dropout percentage
    """

    def __init__(self, p: float):
        super().__init__()
        self.dropout = torch.nn.Dropout(p)
        self.fc1 = torch.nn.Linear(3258, 250)
        self.fc2 = torch.nn.Linear(250, 512)
        self.fc3 = torch.nn.Linear(512, 250)
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Infer the sign from the recordings."""
        x = self.fc1(x)
        y = self.relu(x)
        y = self.dropout(y)
        y = self.fc2(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.fc3(y)

        return x + y


class ASLData(torch.utils.data.Dataset):
    """
    A custom dataset for the ASL data.

    :param datax: The converted landmark data.
    :param datay: The label for the sign.
    """

    def __init__(self, datax: torch.Tensor, datay: torch.Tensor):
        self.datax = (datax - datax.mean(axis=0)) / datax.std(axis=0)
        self.datay = datay

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the item for a given index.

        :param idx: the index of the sample that should be reutrned.
        """
        return self.datax[idx, :], self.datay[idx]

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.datay)


class ASLInferModel(tf.Module):
    """
    Model for the tf-lite method.

    :param tf_feat_gen_path: The path to the stored feature generator.
    :param tf_model_path: The path to the stored classification model.
    """

    def __init__(
        self,
        tf_feat_gen_path: pathlib.Path,
        tf_model_path: pathlib.Path,
    ):
        super().__init__()
        self.feature_gen = tf.saved_model.load(tf_feat_gen_path)
        self.model = tf.saved_model.load(tf_model_path)
        self.feature_gen.trainable = False
        self.model.trainable = False

    @tf.function(
        input_signature=[
            tf.TensorSpec(
                shape=[None, 543, 3], dtype=tf.float32, name="inputs",
            ),
        ],
    )
    def call(
        self, input_tensor: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        """
        Inference function of the model.

        :param input_tensor: The input to the model
        """
        output_tensors = {}
        features = self.feature_gen(input = input_tensor)["output"]
        output_tensors["outputs"] = self.model(
            input = tf.expand_dims(features, 0),
        )["output"][0, :]
        return output_tensors


def load_relevant_data_subset(
    pq_path: str,
    rows_per_frame: int = 543,
) -> npt.NDArray[np.float32]:
    """Load the parquet file."""
    data_columns = ["x", "y", "z"]
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / rows_per_frame)
    data = data.to_numpy().reshape(n_frames, rows_per_frame, len(data_columns))
    return data.astype(np.float32)


def convert_and_save_data(
    train_file: pathlib.Path,
    label_map: dict[str, int],
    feature_converter: FeatureGen,
    landmark_files_dir: pathlib.Path,
    data_file_path: pathlib.Path,
    label_file_path: pathlib.Path,
) -> tuple[npt.NDArray[npt.float64], npt.NDArray[npt.float64]]:
    """Load the data, convert it and store it as a numpy array."""
    dataframe = pd.read_csv(train_file)
    dataframe["label"] = dataframe["sign"].map(label_map)
    npdata = np.zeros((dataframe.shape[0], 3258))
    nplabels = np.zeros(dataframe.shape[0])

    def convert_row(
        row: tuple[str, ...],
    ) -> tuple[npt.NDArray[np.float64], int]:
        """Convert a row of the train file into a sample and label."""
        x = load_relevant_data_subset(landmark_files_dir / row[1].path)
        x = feature_converter(torch.tensor(x)).cpu().numpy()
        return x, row[1].label

    with mp.Pool() as pool:
        results = pool.imap(convert_row, dataframe.iterrows(), chunksize=250)
        for i, (x, y) in tqdm.tqdm(
            enumerate(results),
            total=dataframe.shape[0],
        ):
            npdata[i, :] = x
            nplabels[i] = y

    np.save(data_file_path, npdata)
    np.save(label_file_path, nplabels)

    return npdata, nplabels


def train(
    trainx: npt.NDArray[np.float32],
    trainy: npt.NDArray[np.float32],
    testx: npt.NDArray[np.float32],
    testy: npt.NDArray[np.float32],
    epochs: int = 42,
    t_0: int = 6,
    t_mult: int = 2,
    batch_size: int = 64,
    learning_rate: float = 0.005,
    p: float = 0.5,
) -> torch.nn.Module:
    """
    Train a neural network.

    :param trainx: the training set input features to the neural network.
    :param trainy: the labels for the training set
    :param testx: the test set inputs to the neural network
    :param testy: the test labels
    :param epochs: the number of epochs for which the neural network should be
        trained
    :param T_0: Parameter for the Scheduler
    :param T_mult: Parameter for the scheduler
    :param batch_size: Number of samples in one batch.
    :param learning_rate: The learning rate to scale the gradients in the
        gradient descent.
    :param p: The dropout percentage
    """
    train_data = ASLData(trainx, trainy)
    test_data = ASLData(testx, testy)

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
    )

    model = ASLModel(p)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=t_0,
        T_mult=t_mult,
    )

    for i in range(epochs):
        model.train()

        train_loss_sum = 0
        train_correct = 0
        train_total = 0
        train_bar = tqdm.tqdm(train_loader)
        train_bar.set_description(f"Epoch {i}")
        for batch_np, label_np in train_bar:
            optimizer.zero_grad()

            batch = torch.Tensor(batch_np).float()
            label = torch.Tensor(label_np).long()

            output = model(batch)

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            train_loss_sum = train_loss_sum + loss.item()
            train_correct = train_correct + np.sum(
                np.argmax(output.detach().cpu().numpy(), axis=1)
                == label.cpu().numpy(),
            )
            train_total = train_total + 1
            train_bar.set_postfix(
                {
                    "loss": train_loss_sum / (train_total * batch_size),
                    "acc": train_correct / len(train_data),
                },
            )

        scheduler.step()

    test_loss_sum = 0
    test_correct = 0
    test_total = 0
    model.eval()
    test_bar = tqdm.tqdm(test_loader)
    test_bar.set_description("Test")
    for batch_np, label_np in test_bar:
        batch = torch.Tensor(batch_np).float()
        label = torch.Tensor(label_np).long()
        output = model(batch)

        loss = criterion(output, label)

        test_loss_sum = test_loss_sum + loss.item()
        test_correct = test_correct + np.sum(
            np.argmax(output.detach().cpu().numpy(), axis=1)
            == label.cpu().numpy(),
        )
        test_total = test_total + 1
        test_bar.set_postfix(
            {"loss:": test_loss_sum / (test_total * batch_size)},
        )

    print(
        f"Train Loss: {(train_loss_sum/train_total):.04f},"
        f" Train Acc: {train_correct/len(train_data):0.04f}",
    )
    print(
        f"Val Loss: {(test_loss_sum/test_total):.04f},"
        f" Val Acc: {test_correct/len(test_data):0.04f}",
    )

    return model


def create_submission(
    onnx_feature_gen_path: pathlib.Path,
    tf_feat_gen_path: pathlib.Path,
    feature_converter: torch.nn.Module,
    onnx_asl_model_path: pathlib.Path,
    tf_model_path: pathlib.Path,
    model: torch.nn.Module,
) -> None:
    """
    Create a submission.

    :param onnx_featuere_gen_path: The path where the feature gen should be
        saved.
    :param tf_feat_gen_path: The path where the tf featuer generator is stored.
    :param feature_converter: The feature generator that should be saved.
    :param onnx_asl_model_path: The Path where the model should be stored.
    :param tf_model_path: The path where the tf model should be shared.
    :param model: The model that should be incuded in the tflite.
    """
    sample_input_feature_gen = torch.rand((50, 543, 3))
    sample_input_model = torch.rand((1, 3258))

    feature_converter.eval()
    model.eval()

    torch.onnx.export(
        feature_converter,
        sample_input_feature_gen,
        onnx_feature_gen_path,
        opset_version=12,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "input"}},
    )

    torch.onnx.export(
        model,
        sample_input_model,
        onnx_asl_model_path,
        opset_version=12,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "input"}},
    )

    onnx_feat_gen = onnx.load(onnx_feature_gen_path)
    tf_rep = onnx_tf.backend.prepare(onnx_feat_gen)
    tf_rep.export_graph(tf_feat_gen_path)

    onnx_model = onnx.load(onnx_asl_model_path)
    tf_rep = onnx_tf.backend.prepare(onnx_model)
    tf_rep.export_graph(tf_model_path)


def make_submission(
    onnx_feature_gen_path: pathlib.Path,
    tf_feat_gen_path: pathlib.Path,
    feature_converter: torch.nn.Module,
    onnx_asl_model_path: pathlib.Path,
    tf_model_path: pathlib.Path,
    model: torch.nn.Module,
    tf_lite_path: pathlib.Path,
) -> None:
    """
    Create a submission.

    :param onnx_featuere_gen_path: The path where the feature gen should be
        saved.
    :param tf_feat_gen_path: The path where the tf featuer generator is stored.
    :param feature_converter: The feature generator that should be saved.
    :param onnx_asl_model_path: The Path where the model should be stored.
    :param tf_model_path: The path where the tf model should be shared.
    :param model: The model that should be incuded in the tflite.
    :param tf_lite_path: The directory that should contain the final tf lite
        model
    """
    create_submission(
        onnx_feature_gen_path=onnx_feature_gen_path,
        tf_feat_gen_path=tf_feat_gen_path,
        feature_converter=feature_converter,
        onnx_asl_model_path=onnx_asl_model_path,
        tf_model_path=tf_model_path,
        model=model,
    )

    print(tf_feat_gen_path, tf_model_path)
    mytfmodel = ASLInferModel(
        tf_feat_gen_path,
        tf_model_path,
    )
    tf.saved_model.save(
        mytfmodel,
        tf_path_combined,
        signatures={"serving_default": mytfmodel.call},
    )
    converter = tf.lite.TFLiteConverter.from_saved_model(
        tf_path_combined.as_posix(),
    )
    tflite_model = converter.convert()

    with tf_lite_path.open("wb") as file_pointer:
        file_pointer.write(tflite_model)


if __name__ == "__main__":
    # define some constants
    landmark_files_dir_ = pathlib.Path(
        "/home/reimers/Documents/Projects/ALS/effective-octo-potato/data/",
    )
    train_file_ = pathlib.Path(
        "/home/reimers/Documents/Projects/ALS/effective-octo-potato/data/"
        "train.csv",
    )
    test_file = pathlib.Path(
        "/home/reimers/Documents/Projects/ALS/effective-octo-potato/data/"
        "test.csv",
    )
    label_map_ = json.load(
        pathlib.Path(
            "/home/reimers/Documents/Projects/ALS/effective-octo-potato/data/"
            "sign_to_prediction_index_map.json",
        ).open("r"),
    )
    train_feature_path = pathlib.Path("feature_data.npy")
    train_label_path = pathlib.Path("feature_labels.npy")
    test_feature_path = pathlib.Path("feature_data_test.npy")
    test_label_path = pathlib.Path("featuer_labels_test.npy")

    onnx_feat_gen_path = pathlib.Path("feature_gen.onnx")
    onnx_model_path = pathlib.Path("asl_model.onnx")

    tf_feature_gen_path_ = pathlib.Path("tf_feature_gen")
    tf_model_path_ = pathlib.Path("tf_model")
    tf_path_combined = pathlib.Path("tf_infer_model")
    tf_lite_path = pathlib.Path("model.tflite")

    feature_converter_ = FeatureGen()

    try:
        trainx = np.load(train_feature_path)
        trainy = np.load(train_label_path)
    except OSError:
        trainx, trainy = convert_and_save_data(
            train_file_,
            label_map_,
            feature_converter_,
            landmark_files_dir_,
            train_feature_path,
            train_label_path,
        )
    try:
        testx = np.load(test_feature_path)
        testy = np.load(test_label_path)
    except OSError:
        testx, testy = convert_and_save_data(
            test_file,
            label_map_,
            feature_converter_,
            landmark_files_dir_,
            test_feature_path,
            test_label_path,
        )

    model = train(
        trainx,
        trainy,
        testx,
        testy,
        learning_rate=0.0003,
        epochs=70,
        p=0.5,
    )

    gc.collect()

    make_submission(
        onnx_feature_gen_path=onnx_feat_gen_path,
        tf_feat_gen_path=tf_feature_gen_path_,
        feature_converter=feature_converter_,
        onnx_asl_model_path=onnx_model_path,
        tf_model_path=tf_model_path_,
        model=model,
        tf_lite_path=tf_lite_path,
    )
