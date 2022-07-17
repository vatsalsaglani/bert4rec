from matplotlib.pyplot import hist
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os
import re
from bert4rec_model import RecommendationTransformer
from constants import TRAIN_CONSTANTS
from typing import List, Dict, Tuple
import random

T.manual_seed(3007)
T.cuda.manual_seed(3007)


class Recommender:
    """Recommender Object
    """
    def __init__(self, model_path: str):
        """Recommender object to predict sequential recommendation

        Args:
            model_path (str): Path to the model
        """
        self.model = RecommendationTransformer(
            vocab_size=TRAIN_CONSTANTS.VOCAB_SIZE,
            heads=TRAIN_CONSTANTS.HEADS,
            layers=TRAIN_CONSTANTS.LAYERS,
            emb_dim=TRAIN_CONSTANTS.EMB_DIM,
            pad_id=TRAIN_CONSTANTS.PAD,
            num_pos=TRAIN_CONSTANTS.HISTORY)

        state_dict = T.load(model_path, map_location="cpu")

        self.model.load_state_dict(state_dict["state_dict"])

        self.model.eval()

        self.max_length = 25

    def predict(self, inp_tnsr: T.LongTensor, mode="post"):
        """Predict and return next or prev item in the sequence based on the mode

        Args:
            inp_tnsr (T.LongTensor): Input Tensor of items in the sequence
            mode (str, optional): Predict the start or end item based on the mode. Defaults to "post".

        Returns:
            int: Item ID
        """
        with T.no_grad():
            op = self.model(inp_tnsr.unsqueeze(0), None)
        _, pred = op.max(1)
        if mode == "post":
            pred = pred.flatten().tolist()[-1]
        elif mode == "pre":
            pred = pred.flatten().tolist()[0]
        else:
            pred = pred.flatten().tolist()[-1]

        return pred

    def recommendPre(self, sequence: List[int], num_recs: int = 5):
        """Predict item at start

        Args:
            sequence (List[int]): Input list of items
            num_recs (int, optional): Total number of items to predict. Defaults to 5.

        Returns:
            Tuple: Returns the sequence and history if more predictions than max length
        """
        history = []
        predict_hist = 0
        while predict_hist < num_recs:
            if len(sequence) > TRAIN_CONSTANTS.HISTORY - 1:
                history.extend(sequence)
                sequence = sequence[:TRAIN_CONSTANTS.HISTORY - 1]
            inp_seq = T.LongTensor(sequence)
            inp_tnsr = T.ones((inp_seq.size(0) + 1), dtype=T.long)
            inp_tnsr[1:] = inp_seq
            pred = self.predict(inp_tnsr, mode="pre")
            sequence = [pred] + sequence
            predict_hist += 1

        return sequence, history

    def recommendPost(self, sequence: List[int], num_recs: int = 5):
        """Predict item at end

        Args:
            sequence (List[int]): Input list of items
            num_recs (int, optional): Total number of item to predict. Defaults to 5.

        Returns:
            Tuple: Returns the sequence and history if more predictions than max length
        """
        history = []
        predict_hist = 0
        while predict_hist < num_recs:
            if len(sequence) > TRAIN_CONSTANTS.HISTORY - 1:
                history.extend(sequence)
                sequence = sequence[::-1][:TRAIN_CONSTANTS.HISTORY - 1][::-1]
            inp_seq = T.LongTensor(sequence)
            inp_tnsr = T.ones((inp_seq.size(0) + 1), dtype=T.long)
            inp_tnsr[:inp_seq.size(0)] = inp_seq
            pred = self.predict(inp_tnsr)
            sequence.append(pred)
            predict_hist += 1

        return sequence, history

    def recommendSequential(self, sequence: List[int], num_recs: int = 5):
        """Predicts both start and end items randomly

        Args:
            sequence (List[int]): Input list of items
            num_recs (int, optional): Total number of items to predict. Defaults to 5.

        Returns:
            Tuple: Returns the sequence and history (empty always)
        """
        assert num_recs < (
            self.max_length / 2
        ) - 1, f"Can only recommend: {num_recs < (self.max_length / 2) - 1} with sequential recommendation"

        history = []
        predict_hist = 0
        while predict_hist < num_recs:
            if bool(random.choice([0, 1])):
                # print(f"RECOMMEND POST")
                sequence, hist = self.recommendPost(sequence, 1)
                # print(f"SEQUENCE: {sequence}")
                if len(hist) > 0:
                    history.extend(hist)
            else:
                # print(f"RECOMMEND PRE")
                sequence, hist = self.recommendPre(sequence, 1)
                # print(f"SEQUENCE: {sequence}")
                if len(hist) > 0:
                    history.extend(hist)
            predict_hist += 1

        return sequence, []

    def cleanHistory(self, history: List[int]):
        """History might have multiple repetitions, we clean the history 
        and maintain the sequence

        Args:
            history (List[int]): Predicted item ids

        Returns:
            List[int]: Returns cleaned item id
        """
        history = history[::-1]
        history = [
            h for ix, h in enumerate(history) if h not in history[ix + 1:]
        ]
        return history[::-1]

    def recommend(self,
                  sequence: List[int],
                  num_recs: int = 5,
                  mode: str = "post"):
        """Recommend Items

        Args:
            sequence (List[int]): Input list of items
            num_recs (int, optional): Total number of items to predict. Defaults to 5.
            mode (str, optional): Predict start or end items or creates a random sequence around the input sequence. Defaults to "post".

        Returns:
            List[int]: Recommended items
        """
        if mode == "post":

            seq, hist = self.recommendPost(sequence, num_recs)

        elif mode == "pre":

            seq, hist = self.recommendPre(sequence, num_recs)

        else:

            seq, hist = self.recommendSequential(sequence, num_recs)

        hist = self.cleanHistory(hist)

        if len(hist) > 0 and len(hist) > len(seq):
            return hist

        return seq
