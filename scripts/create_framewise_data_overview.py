""" Create Framewise Data Overview.

The goal of the script is:
    1. Modify/alter the original train.csv from the challenge data.
        - Modification/alternative such that train.csv table contains columns
          describing frame range for each parquet file.
            - Intention is to access single frames in torch DataLoader.
            - Disadvantage is loading whole parquet file for single frame.
            - Advantage is better control over batch size.
"""

import pathlib

import pandas as pd 
import tqdm


if __name__ == "__main__":
    file_name_altered = 'train_frames.csv'
    file_name_original = "ori_train.csv"
    data_base_path = pathlib.Path(__file__).parent.parent / "data"
    
    # load data
    train_file = data_base_path / file_name_original
    train_df = pd.read_csv(train_file)

    # extract [min, max] frame ranges
    def _get_frame_range(df: pd.DataFrame) -> tuple[int, int]:
        frames: pd.Series = df['frame']
        return frames.min(), frames.max()
    
    frame_ranges: list[tuple[int, int]] = [
        _get_frame_range(pd.read_parquet(data_base_path / parquet_file_name)) 
        for parquet_file_name in tqdm.tqdm(train_df['path'])
    ]
    min_frames, max_frames = zip(*frame_ranges)
    
    # create modified table
    train_df['frame_min'] = min_frames
    train_df['frame_max'] = max_frames

    train_df.to_csv(data_base_path / file_name_altered, index=False)
