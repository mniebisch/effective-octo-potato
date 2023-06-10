import json
import pathlib
import warnings
from typing import Any, List, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch_geometric.loader as pyg_loader
import torch_geometric.transforms as pyg_transforms
import tqdm
from sklearn import model_selection

import wandb
from effective_octo_potato.models import GCN
from effective_octo_potato.temporal_feature_generator import (
    TemporalFeatureGenerator,
    create_pyg_dataset,
    handle_training_data,
)
from effective_octo_potato.transforms import (
    CalcReferenceFeatures,
    CatNodeFeatures,
    DropDepthDimension,
    PosSplitNodes,
    PosStackNodes,
)
from effective_octo_potato.utils import load_labels

# TODO 1 up-/or downsampling of frames to fixed number of frames via interpolation [IMPLEMENTED]
# TODO 2 could a kinda transformer handle variable size of frames?
# TODO 3 create huge graph reflecting temporal patterns (not compatible with TODO 2) [IMPLEMENTED]
# TODO 4 use/create edge labels for temporal pattern graph in TODO 3
# TODO 5 join latest step in FeatureGenerator [IMPLEMENTED]
# TODO 6 consider adding timestep indicator similar to pyg's batch variable (returned from pyg.DataLoader) [IMPLEMENTED]
# TODO 7 consider adding node indicator similar to pyg's batch variable (returned from pyg.DataLoader) [IMPLEMENTED]
# TODO 8 make reference nodes aditional output. remove distance from node features. [IMPLEMENTED]
# compute later in augmentation. add time_step_indices information for reference nodes.

# https://github.com/pytorch/pytorch/issues/97207
warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)


def get_data_split(
    group_labels: pd.Series, train_size: float = 0.8, is_submission: bool = True
) -> Tuple[npt.NDArray[np.integer], npt.NDArray[np.integer]]:
    if is_submission:
        raise NotImplementedError

    splitter = model_selection.GroupShuffleSplit(n_splits=1, train_size=train_size)
    group_split = splitter.split(None, None, group_labels)
    return next(group_split)


def eval(
    model: torch.nn.Module,
    dataloader: pyg_loader.DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        batch_iterator = tqdm.tqdm(dataloader)
        for data in batch_iterator:
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            edge_attr = data.edge_attr.to(device)
            batch = data.batch.to(device)
            prediction = model(x, edge_index, edge_attr, batch)
            target = torch.tensor(data.y, dtype=torch.long)
            prediction_label = torch.argmax(prediction, dim=1)
            prediction_label = prediction_label.detach().cpu()
            total += target.shape[0]
            correct += (target == prediction_label).sum().item()
    return correct / total


if __name__ == "__main__":
    wandb.init(project="asl-challenge")
    data_base_path = pathlib.Path(__file__).parent.parent / "data"
    output_base_path = pathlib.Path(__file__).parent.parent / "data"

    is_submission = False

    ckpt_path = data_base_path / "ckpts"
    ckpt_name = "some_ckpt.tar"
    ckpt_file = ckpt_path / ckpt_name

    # load/process labels
    data_csv = "train.csv"
    train_df = pd.read_csv(data_base_path / data_csv)
    labels = load_labels(data_dir=data_base_path, labels=train_df["sign"])

    # create/load features
    # feature_matrix_file_name = "graph_data_meanstd_subpose_reference_nodes.zip"
    feature_matrix_file_name = "first_temporal_graph_with_edge_attr.zip"
    fg = TemporalFeatureGenerator()

    feature_matrix = handle_training_data(
        data=train_df,
        raw_dir=data_base_path,
        feature_dir=output_base_path,
        feature_generator=fg,
        feature_file_name=feature_matrix_file_name,
    )

    # split data
    train_indices, valid_indices = get_data_split(
        group_labels=train_df["participant_id"], is_submission=is_submission
    )
    train_indices = train_indices.tolist()
    valid_indices = valid_indices.tolist()

    def _indexing_helper(data: List[Any], indices: List[int]) -> list[Any]:
        return [data[ind] for ind in indices]

    train_data = _indexing_helper(feature_matrix, train_indices)
    valid_data = _indexing_helper(feature_matrix, valid_indices)
    train_labels = labels[train_indices]
    valid_labels = labels[valid_indices]

    # transforms
    train_transform = pyg_transforms.Compose(
        [
            PosStackNodes(),
            pyg_transforms.NormalizeScale(),
            # pyg_transforms.RandomShear(shear=0.01),
            pyg_transforms.RandomJitter(translate=0.005),
            pyg_transforms.RandomRotate(degrees=25, axis=0),
            pyg_transforms.RandomRotate(degrees=25, axis=1),
            pyg_transforms.RandomRotate(degrees=25, axis=2),
            PosSplitNodes(),
            DropDepthDimension(),
            CalcReferenceFeatures(),
            CatNodeFeatures(),
        ]
    )
    valid_transform = pyg_transforms.Compose(
        [
            PosStackNodes(),
            pyg_transforms.NormalizeScale(),
            PosSplitNodes(),
            DropDepthDimension(),
            CalcReferenceFeatures(),
            CatNodeFeatures(),
        ]
    )

    train_graphs = create_pyg_dataset(
        train_data, train_labels, transform=train_transform
    )
    valid_graphs = create_pyg_dataset(
        valid_data, valid_labels, transform=valid_transform
    )

    # hyperparams
    batch_size = 128  # 128
    epochs = 175

    # pyg stuff
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dataloader = pyg_loader.DataLoader(
        train_graphs,
        batch_size=batch_size,
        shuffle=True,
        num_workers=10,
        drop_last=True,
    )
    valid_dataloader = pyg_loader.DataLoader(
        valid_graphs, batch_size=batch_size, shuffle=False, num_workers=10
    )
    num_feature_nodes = train_graphs[0].num_node_features
    hidden_channels = 512
    num_blocks = 0
    model = GCN(
        num_node_features=num_feature_nodes,
        hidden_channels=hidden_channels,  # 512,  # 128,
        num_classes=250,
        num_blocks=num_blocks,
        drop_p=0.0,
    )
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5 * 10e-5, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=25,
        T_mult=2,
        verbose=True,
    )

    wandb.watch(model, log="all")
    model.train()
    for epoch_idx in range(epochs):
        batch_iterator = tqdm.tqdm(
            train_dataloader, desc=f"Epoch: {epoch_idx+1:02d}/{epochs}"
        )
        rolling_loss = None
        for data in batch_iterator:
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            edge_attr = data.edge_attr.to(device)
            batch = data.batch.to(device)
            optimizer.zero_grad()
            prediction = model(x, edge_index, edge_attr, batch)
            target = torch.tensor(data.y, dtype=torch.long)
            target = target.to(device)
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()
            if rolling_loss is None:
                rolling_loss = loss.item()
            else:
                rolling_loss = 0.9 * rolling_loss + 0.1 * loss.item()
            batch_iterator.set_postfix({"loss": rolling_loss})
            wandb.log({"loss": loss.item()})

        current_lr = scheduler.get_last_lr()[0]
        wandb.log({"fu": current_lr})

        if epoch_idx % 10 == 0:
            train_acc = eval(model, train_dataloader, device)
            valid_acc = eval(model, valid_dataloader, device)
            wandb.log({"train": {"acc": train_acc}, "val": {"acc": valid_acc}})

        scheduler.step()

    # eval
    train_acc = eval(model, train_dataloader, device)
    valid_acc = eval(model, valid_dataloader, device)
    print("train acc", train_acc)
    print("valid acc", valid_acc)

    # store data
    torch.save({"model_state_dict": model.state_dict()}, ckpt_file)
    eval_json_path = data_base_path / "eval_jsons"
    json_name = "some_data.json"
    params = {
        "data": {
            "feature_matrix_file_name": feature_matrix_file_name,
            "ckpt_name": ckpt_name,
        },
        "indices": {
            "train": train_indices,
            "valid": valid_indices,
        },
        "params": {
            "num_feature_nodes": num_feature_nodes,
            "hidden_channels": hidden_channels,
            "num_blocks": num_blocks,
        },
    }
    with open(eval_json_path / json_name, "w") as file:
        json.dump(params, file, indent=4)
