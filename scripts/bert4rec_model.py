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
                 num_pos=128,
                 num_channels=128):
        """Recommendation model initializer

        Args:
            vocab_size (int): Number of unique tokens/items
            heads (int, optional): Number of heads in the Multi-Head Self Attention Transformers (). Defaults to 4.
            layers (int, optional): Number of Layers. Defaults to 6.
            emb_dim (int, optional): Embedding Dimension. Defaults to 512.
            pad_id (int, optional): Token used to pad tensors. Defaults to 0.
            num_pos (int, optional): Positional Embedding, fixed sequence. Defaults to 120.
            num_channels (int, optional): Channels for Item Embeddings
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.pad_id = pad_id
        self.num_pos = num_pos
        self.vocab_size = vocab_size
        print(f'NUM POS: ', num_pos)
        self.item_embeddings = nn.Embedding(self.vocab_size,
                                            embedding_dim=num_channels)
        self.encoder = Encoder(
            source_vocab_size=num_channels,
            emb_dim=num_channels,
            layers=layers,
            heads=heads,
            dim_key=num_channels,
            dim_value=num_channels,
            dim_model=num_channels,
            dim_inner=num_channels * 4,
            pad_id=pad_id,
            num_pos=num_pos,
        )
        self.lin_op = nn.Linear(num_channels, self.vocab_size)

    def forward(self, batch):
        """Returns predictions for a given sequence of tokens/items

        Args:
            batch (torch.Tensor): Batch of sequences

        Returns:
            torch.Tensor: Prediction for masked items
        """
        # batch_size = batch.size(0)
        # print(f'BATCH: {batch.size()}')
        src_items = self.item_embeddings(batch)
        # print(f'ITEMS: {src_items.size()}')
        batch_size, in_sequence_len = src_items.size(0), src_items.size(1)
        batch = T.arange(0, in_sequence_len,
                         device=src_items.device).unsqueeze(0).repeat(
                             batch_size, 1)
        # print(f'REPEAT BATCH: {batch.size()}')
        op = self.encoder(batch, None)
        # print(f'ENCODED BATCH: {op.size()}')
        # op = op.view(-1, op.size(2))
        op = op.permute(1, 0, 2)
        # print(f'PERMUTED BATCH: {op.size()}')
        op = self.lin_op(op)
        # print(f'AFTER LINEAR BATCH: {op.size()}')
        # op = op.view(batch_size, int(op.size(0) / batch_size), -1)
        return op
