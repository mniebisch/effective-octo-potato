import pathlib

import pandas as pd

if __name__ == "__main__":
    data_base_path = pathlib.Path(__file__).parent.parent / "data"

    # load labels
    data_csv = "train.csv"
    train_df = pd.read_csv(data_base_path / data_csv)
    example_data = pd.read_parquet(data_base_path / train_df["path"][1])
    first_frame = example_data[:543]

    # body part node indices
    lip = [
        61,
        185,
        40,
        39,
        37,
        0,
        267,
        269,
        270,
        409,
        291,
        146,
        91,
        181,
        84,
        17,
        314,
        405,
        321,
        375,
        78,
        191,
        80,
        81,
        82,
        13,
        312,
        311,
        310,
        415,
        95,
        88,
        178,
        87,
        14,
        317,
        402,
        318,
        324,
        308,
    ]
    left_hand = list(range(468, 489))
    right_hand = list(range(522, 543))
    pose = list(range(489, 522))
    face = list(range(0, 468))

    print(example_data)
