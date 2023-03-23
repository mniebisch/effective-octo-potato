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

import matplotlib.pyplot as plt
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
        self.output_shape = 3258

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

class FeatureGen2(torch.nn.Module):
    """
    Convert the features (n_frames, 543, 3) into (n_features,).
    
    In this feature gen, we heavyly reduce the number of landmarks.
    """

    def __init__(self):
        super().__init__()
        self.output_shape = 270

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert the features."""
        face_x = x[:, :468, :].contiguous()
        face_x = torch.mean(face_x, dim = 1)
        pose_x = x[:, (489 + 23, 489 + 24), :].contiguous().view(-1, 2 * 3)
        lefth_x = x[:, 468:489, :].contiguous().view(-1, 21 * 3)
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

class FeatureGen3(torch.nn.Module):
    """
    Convert the features (n_frames, 543, 3) into (n_features,).
    
    In this feature gen, we heavyly reduce the number of landmarks.
    """

    def __init__(self):
        super().__init__()
        self.output_shape = 639

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert the features."""
        face_x = x[:, :468, :].contiguous()
        pose_x = x[:, (489 + 23, 489 + 24), :].contiguous().view(-1, 2 * 3) 
        x1m = torch.mean(face_x, (0, 1))
        x3m = torch.mean(pose_x, 0)

        nr_frames = x.shape[0]
        frames = (0, (nr_frames - 1) / 4, (nr_frames-1) / 2, (nr_frames-1) * 3 / 4, nr_frames-1)

        lefth_x = x[:, 468:489, :]
        lefth_0 = lefth_x[0]
        lefth_1 = (frames[1] - int(frames[1])) * lefth_x[int(frames[1])] + (frames[1] -1 - int(frames[1])) * lefth_x[int(frames[1]) + 1]
        lefth_2 = (frames[2] - int(frames[2])) * lefth_x[int(frames[2])] + (frames[2] -1 - int(frames[2])) * lefth_x[int(frames[2]) + 1]
        lefth_3 = (frames[3] - int(frames[3])) * lefth_x[int(frames[3])] + (frames[3] -1 - int(frames[3])) * lefth_x[int(frames[3]) + 1]
        lefth_4 = lefth_x[-1]

        lefth_x = torch.concatenate([lefth_0, lefth_1, lefth_2, lefth_3, lefth_4], axis = 0)
        lefth_x_anker = lefth_x[:,0:1]
        lefth_x = lefth_x - lefth_x_anker
        lefth_x[:,0:1] = lefth_x_anker
        lefth_x = lefth_x.contiguous().view(15 * 21)



        righth_x = x[:, 522:, :]
        righth_0 = righth_x[0]
        righth_1 = (frames[1] - int(frames[1])) * righth_x[int(frames[1])] + (frames[1] -1 - int(frames[1])) * righth_x[int(frames[1]) + 1]
        righth_2 = (frames[2] - int(frames[2])) * righth_x[int(frames[2])] + (frames[2] -1 - int(frames[2])) * righth_x[int(frames[2]) + 1]
        righth_3 = (frames[3] - int(frames[3])) * righth_x[int(frames[3])] + (frames[3] -1 - int(frames[3])) * righth_x[int(frames[3]) + 1]
        righth_4 = righth_x[-1]
        
        righth_x = torch.concatenate([righth_0, righth_1, righth_2, righth_3, righth_4], axis = 0)
        righth_x_anker = righth_x[:,0:1]
        righth_x = righth_x - righth_x_anker
        righth_x[:,0:1] = righth_x_anker
        righth_x = righth_x.contiguous().view(15 * 21)

 
        xfeat = torch.cat([x1m, x3m, lefth_x, righth_x], axis=0)
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

    def __init__(self, 
            p: float, 
            nr_inputs: int = 3258,
            mean: torch.Tensor | None = None,
            inv_std: torch.Tensor | None = None,
            nr_of_blocks: int = 3,
            size: int = 512,
    ):
        super().__init__()

        if mean is None:
            mean = torch.zeros(nr_inputs)
        if inv_std is None:
            inv_std = torch.ones(nr_inputs)
            
        self.mean = mean.unsqueeze(0)
        self.inv_std = inv_std.unsqueeze(0)

        self.dropout = torch.nn.Dropout(p)
        self.fc1 = torch.nn.Linear(nr_inputs, size)
        self.block_layer = torch.nn.ModuleList([
            torch.nn.Linear(size, 2 * size),
            torch.nn.Linear(2 * size, size),
            ] * nr_of_blocks)
        
        self.final_layer = torch.nn.Linear(size, 250)
        self.relu = torch.nn.ReLU()
        self.nr_blocks = nr_of_blocks


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Infer the sign from the recordings."""
        

        x = (x - self.mean) * self.inv_std

        x = self.fc1(x)
        
        for i in range(self.nr_blocks):
            y = self.relu(x)
            y = self.dropout(y)
            y = self.block_layer[2 * i](y)
            y = self.relu(y)
            y = self.dropout(y)
            y = self.block_layer[2 * i + 1](y)

            x = x + y
        x = self.relu(x)

        return self.final_layer(x)


class ASLData(torch.utils.data.Dataset):
    """
    A custom dataset for the ASL data.

    :param datax: The converted landmark data.
    :param datay: The label for the sign.
    """

    def __init__(self, datax: torch.Tensor, datay: torch.Tensor):
        self.datax = datax
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
    npdata = np.zeros((dataframe.shape[0], feature_converter.output_shape))
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
    valx: npt.NDArray[np.float32],
    valy: npt.NDArray[np.float32],
    model: torch.nn.Module,
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
    :param valx: the features of the validation samples
    :param valy: the labels for the validation samples
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
    val_data = ASLData(valx, valy)

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size = batch_size,
        num_workers = 0,
        shuffle = False,
    )

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
        
        val_loss_sum = 0
        val_correct = 0
        val_total = 0
        val_bar = tqdm.tqdm(val_loader)
        val_bar.set_description(f"Epoch {i}")
        for batch_np, label_np in val_bar:
            batch = torch.Tensor(batch_np).float()
            label = torch.Tensor(label_np).long()

            output = model(batch)

            loss = criterion(output, label)

            val_loss_sum = val_loss_sum + loss.item()
            val_correct = val_correct + np.sum(
                np.argmax(output.detach().cpu().numpy(), axis=1)
                == label.cpu().numpy(),
            )
            val_total = val_total + 1
            val_bar.set_postfix(
                {
                    "val loss": val_loss_sum / (val_total * batch_size),
                    "val acc": val_correct / len(val_data),
                },
            )



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
    nr_inputs: int = 3258,
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
    sample_input_model = torch.rand((1, nr_inputs))

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
    nr_inputs: int = 3258,
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
        nr_inputs = nr_inputs,
    )

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

def create_confusion(model, trainx, trainy, testx, testy, 
    batch_size: int = 64,
):

    train_data = ASLData(trainx, trainy)
    test_data = ASLData(testx, testy)

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True,
    )

    confusion_test = np.zeros((250, 250))
    test_bar = tqdm.tqdm(test_loader)
    test_bar.set_description("Test")
    for batch_np, label_np in test_bar:
        batch = torch.Tensor(batch_np).float()
        label = torch.Tensor(label_np).long()
        
        output = model(batch)

        #loss = criterion(output, label)
        #test_loss_sum = test_loss_sum + loss.item()

        pred = np.argmax(output.detach().cpu().numpy(), axis=1)
        label = label.cpu().numpy()
        
        for i in range(pred.shape[0]):
            confusion_test[pred[i], label[i]] += 1

    plt.figure()
    plt.imshow(confusion_test)
    plt.colorbar()
    plt.show()
    plt.close()
    


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
    val_file = pathlib.Path(
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
    test_label_path = pathlib.Path("feature_labels_test.npy")
    val_feature_path = pathlib.Path("feature_data_val.npy")
    val_label_path = pathlib.Path("feature_labels_val.npy")

    onnx_feat_gen_path = pathlib.Path("feature_gen.onnx")
    onnx_model_path = pathlib.Path("asl_model.onnx")

    tf_feature_gen_path_ = pathlib.Path("tf_feature_gen")
    tf_model_path_ = pathlib.Path("tf_model")
    tf_path_combined = pathlib.Path("tf_infer_model")
    tf_lite_path = pathlib.Path("model.tflite")

    feature_converter_ = FeatureGen3()

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
        valx = np.load(val_feature_path)
        valy = np.load(val_label_path)
    except OSError:
        valx, valy = convert_and_save_data(
            val_file,
            label_map_,
            feature_converter_,
            landmark_files_dir_,
            val_feature_path,
            val_label_path,
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

    model = ASLModel(
        p = 0.5, 
        nr_inputs = feature_converter_.output_shape,
        mean = torch.Tensor(trainx.mean(axis = 0)),
        inv_std = torch.Tensor(1 / trainx.std(axis = 0)),
        nr_of_blocks = 100,
        size = 75
    )

    model = train(
        trainx,
        trainy,
        testx,
        testy,
        valx,
        valy,
        model,
        learning_rate=0.0003,
        epochs=186,
        p=0.8,
    )

    create_confusion(model, trainx, trainy, testx, testy)
    
    gc.collect()

    make_submission(
        onnx_feature_gen_path=onnx_feat_gen_path,
        tf_feat_gen_path=tf_feature_gen_path_,
        feature_converter=feature_converter_,
        onnx_asl_model_path=onnx_model_path,
        tf_model_path=tf_model_path_,
        model=model,
        tf_lite_path=tf_lite_path,
        nr_inputs = feature_converter_.output_shape,
    )
