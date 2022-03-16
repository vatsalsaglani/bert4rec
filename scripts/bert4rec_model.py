import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from AttentionTransformer.Encoder import Encoder


class RecommendationModel(nn.Module):
    """Sequential recommendation model architecture
    """
    def __init__(self,
                 vocab_size,
                 heads=4,
                 layers=6,
                 emb_dim=512,
                 pad_id=0,
                 num_pos=120):
        """Recommendation model initializer

        Args:
            vocab_size (int): Number of unique tokens/items
            heads (int, optional): Number of heads in the Multi-Head Self Attention Transformers (). Defaults to 4.
            layers (int, optional): Number of Layers. Defaults to 6.
            emb_dim (int, optional): Embedding Dimension. Defaults to 512.
            pad_id (int, optional): Token used to pad tensors. Defaults to 0.
            num_pos (int, optional): Positional Embedding, fixed sequence. Defaults to 120.
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.pad_id = pad_id
        self.num_pos = num_pos
        self.vocab_size = vocab_size
        print(f'NUM POS: ', num_pos)
        self.encoder = Encoder(
            source_vocab_size=vocab_size,
            emb_dim=emb_dim,
            layers=layers,
            heads=heads,
            dim_key=emb_dim,
            dim_value=emb_dim,
            dim_model=emb_dim,
            dim_inner=emb_dim * 4,
            pad_id=pad_id,
            num_pos=num_pos,
        )
        self.lin_op = nn.Linear(emb_dim, self.vocab_size)

    def forward(self, batch):
        """Returns predictions for a given sequence of tokens/items

        Args:
            batch (torch.Tensor): Batch of sequences

        Returns:
            torch.Tensor: Prediction for masked items
        """
        batch_size = batch.size(0)
        op = self.encoder(batch, None)
        op = op.view(-1, op.size(2))
        op = self.lin_op(op)
        op = op.view(batch_size, int(op.size(0) / batch_size), -1)
        return op
