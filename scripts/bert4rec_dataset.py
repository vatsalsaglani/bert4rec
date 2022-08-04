import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import random
import pandas as pd
from constants import TRAIN_CONSTANTS
from typing import List, Dict, Tuple


class Bert4RecDataset(Dataset):
    """Dataset object for Bert for recommendation

    Args:
        Dataset (torch.utils.data.Dataset)
    """
    def __init__(self,
                 data_csv: pd.DataFrame,
                 group_by_col: str,
                 data_col: str,
                 train_history: int = 120,
                 valid_history: int = 5,
                 padding_mode: str = "right",
                 split_mode: str = "train",
                 threshold_column="rating",
                 threshold=3.5,
                 timestamp_col="timestamp"):
        """Bert4RecDataset object

        Args:
            data_csv (pd.Dataframe): Dataframe containing data
            group_by_col (str): Column name to group the data by
            data_col (str): Column name to fetch the sequence
            train_history (int, optional): Maximum training sequence length. Defaults to 120.
            valid_history (int, optional): Maximum validation sequence length. Defaults to 5.
            padding_mode (str, optional): Pad to left or right. Defaults to right.
            split_mode (str, optional): Split data for train, valid and test. Defaults to train.
            threshold_column (str, optional): Get column with regressive threshold
            threshold (int, optional): Consider rows where threshold_column greater than threshold
            timestamp_col (str, optional): Column name where timestamps are stored
        """
        super().__init__()

        self.data_csv = data_csv
        self.group_by_col = group_by_col
        self.data_col = data_col
        self.train_history = train_history
        self.train_history = train_history
        self.valid_history = valid_history
        self.pad = TRAIN_CONSTANTS.PAD
        self.mask = TRAIN_CONSTANTS.MASK
        self.padding_mode = padding_mode
        self.split_mode = split_mode
        self.timestamp_col = timestamp_col

        if threshold_column:
            self.data_csv = self.data_csv[
                self.data_csv[threshold_column] >= threshold]
            self.data_csv.reset_index(inplace=True)


        self.groups_df = self.data_csv.groupby(by=self.group_by_col)
        self.groups = list(self.groups_df.groups)

    def pad_sequence(self, tokens: List, padding_mode: str = "left"):
        """Pad list to same length

        Args:
            tokens (List): Sequence
            padding_mode (str): Pad to left or right
        Returns:
            List: Sequence
        """
        if len(tokens) < self.train_history:
            if padding_mode == "left":
                tokens = [self.pad
                          ] * (self.train_history - len(tokens)) + tokens
            else:
                tokens = tokens + [self.pad
                                   ] * (self.train_history - len(tokens))

        return tokens

    def get_sequence(self, group_df: pd.DataFrame):
        """Based on the group randomly selects a sequence

        Args:
            group_df (pd.DataFrame): Individual group

        Raises:
            ValueError: If split_mode is not in ["train", "valid", "test"] error will be raised

        Returns:
            pd.DataFrame: Dataframe of length of train or valid history based on the selected split mode
        """
        if self.split_mode == "train":
            _ = group_df.shape[0] - self.valid_history
            end_ix = random.randint(10, _ if _ >= 10 else 10)
        elif self.split_mode in ["valid", "test"]:
            end_ix = group_df.shape[0]
        else:
            raise ValueError(
                f"Split should be either of `train`, `valid`, or `test`. {self.split_mode} is not supported"
            )

        start_ix = max(0, end_ix - self.train_history)

        sequence = group_df[start_ix:end_ix]

        return sequence

    def mask_sequence(self, sequence: List, p: int = 0.8):
        """Randomly mask sequence by replacing correct with MASK token

        Args:
            sequence (List): List of tokens/sequence
            p (int, optional): Mask substitution threshold. Defaults to 0.8.

        Returns:
            List: Masked Sequence
        """
        return [
            s if random.random() < p else TRAIN_CONSTANTS.MASK
            for s in sequence
        ]

    def mask_last_elements_sequence(self, sequence):
        """Only mask the last element/s in the list

        Args:
            sequence (List): List of tokens/sequence

        Returns:
            List: Masked Sequence at end
        """
        sequence = sequence[:-self.valid_history] + self.mask_sequence(
            sequence[-self.valid_history:], p=0.5)
        return sequence

    def get_item(self, idx):
        """Based on the idx get the source and target sequence with their corresponding
        masks

        Args:
            idx (int): Index of group

        Returns:
            Dict: A dict with source, target and their corresponding masks
        """
        group = self.groups[idx]

        group_df = self.groups_df.get_group(group)

        group_df = group_df.sort_values(by=[self.timestamp_col])

        group_df.reset_index(inplace = True)

        sequence = self.get_sequence(group_df)

        trg_items = sequence[self.data_col].tolist()

        if self.split_mode == "train":
            src_items = self.mask_sequence(trg_items)
        else:
            src_items = self.mask_last_elements_sequence(trg_items)

        pad_mode = "left" if random.random() < 0.5 else "right"
        trg_items = self.pad_sequence(trg_items, pad_mode)
        src_items = self.pad_sequence(src_items, pad_mode)

        trg_mask = [1 if t != TRAIN_CONSTANTS.PAD else 0 for t in trg_items]
        src_mask = [1 if t != TRAIN_CONSTANTS.PAD else 0 for t in src_items]

        src_items = T.tensor(src_items, dtype=T.long)
        trg_items = T.tensor(trg_items, dtype=T.long)
        src_mask = T.tensor(src_mask, dtype=T.long)
        trg_mask = T.tensor(trg_mask, dtype=T.long)

        return {
            "source": src_items,
            "target": trg_items,
            "source_mask": src_mask,
            "target_mask": trg_mask
        }

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        return self.get_item(index)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    data = pd.read_csv("../data/ratings_mapped.csv")
    ds = Bert4RecDataset(data_csv=data,
                         group_by_col="userId",
                         data_col="movieId_mapped")
    dl = DataLoader(ds, batch_size=2, shuffle=True)
    tnsr = next(iter(dl))
    print(tnsr)
