import torch as T
import torch.nn.functional as F
from typing import List


def calculate_loss(y_pred: T.tensor, y_true: T.tensor, mask: T.tensor):
    """Calculates the loss between true and predicted using cross entropy

    Args:
        y_pred (T.tensor): Prediction from the model
        y_true (T.tensor): Ground Truth
        mask (T.tensor): Boolean tensor with True for masked tokens and False
        for unmasked

    Returns:
        T.tensor: Total loss
    """
    y_pred = y_pred.view(-1, y_pred.size(2))
    y_true = y_true.view(-1)
    loss = F.cross_entropy(y_pred, y_true, reduction="none")
    loss = loss * mask.view(-1)
    loss = loss.sum() / (mask.sum() + 1e-8)
    return loss


def calculate_accuracy(y_pred: T.tensor, y_true: T.tensor, mask: T.tensor):
    """Calculate the accuracy of the predicted sequence

    Args:
        y_pred (T.tensor): Prediction from the model
        y_true (T.tensor): Ground Truth
        mask (T.tensor): Boolean tensor with True for masked tokens and False
        for unmasked

    Returns:
        T.tensor: Prediction Accuracy
    """
    _, prediction = y_pred.max(2)
    y_true = T.masked_select(y_true, mask)
    prediction = T.masked_select(prediction, mask)

    return (y_true == prediction).double().mean()


def calculate_combined_mean(batch_sizes: List, means: List):
    """Combined Mean of batch accuracy

    Args:
        batch_sizes (List): Number of items in the batch at every iteration
        means (List): Accuracy of every batch prediction
    
    Returns:
        float: Epoch Accuracy
    """

    combined_mean = (T.sum(T.tensor(batch_sizes) * T.tensor(means)) /
                     T.sum(T.tensor(batch_sizes))) * 100
    return combined_mean
