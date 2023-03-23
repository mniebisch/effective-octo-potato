"""
File to split the labels into train and test.

The labels in the train.csv file are split into train and test labels, where
the test.csv contains the examples of signers 61333, 36257, 25571, 37555 and
22343. The new train.csv contains all other signers.

author = Christian Reimers
credit = Team tbd.
"""
from __future__ import annotations 

import csv
import pathlib

import tqdm

def split_train_test(
    original_data_path: pathlib.Path | str = "../data/ori_train.csv",
    train_path: pathlib.Path | str = "../data/train.csv",
    test_path: pathlib.Path | str = "../data/test.csv",
    test_signers: tuple[int, ...] = (61333, 36257, 25571, 37555, 22343),
) -> None:
    """Split the train and the test data."""
    with (
        original_data_path.open("r") as original_data_file,
        train_path.open("w") as train_file,
        test_path.open("w") as test_file,
    ):
        reader = csv.reader(original_data_file)
        train_writer = csv.writer(train_file)
        test_writer = csv.writer(test_file)

        row = next(reader)
        train_writer.writerow(row)
        test_writer.writerow(row)

        for row in tqdm.tqdm(reader, total=94477):
            if int(row[1]) in test_signers:
                test_writer.writerow(row)
            else:
                train_writer.writerow(row)

if __name__ == "__main__":
    original_data_path = pathlib.Path("../../data/ori_train.csv")
    train_path = pathlib.Path("../../data/train_.csv")
    test_path = pathlib.Path("../../data/test.csv")
    train_val_path = pathlib.Path("../../data/train.csv")
    val_path = pathlib.Path("../../data/val.csv")
    test_signers = (61333, 36257, 25571, 37555, 22343)
    val_signers = (26734,)

    split_train_test(
        original_data_path,
        train_path,
        test_path,
        test_signers,
    )

    split_train_test(
            train_path,
            train_val_path,
            val_path,
            val_signers,
    )
