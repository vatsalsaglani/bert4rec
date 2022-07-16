import torch
import torch.nn as nn
from attention import MultiHeadAttention
from embeddings import PositionWiseFeedForward


class EncoderLayer(nn.Module):
    ''' Two Layer Architecture '''
    def __init__(self,
                 dim_model,
                 dim_inner,
                 heads,
                 dim_key,
                 dim_value,
                 dropout=0.1):

        super(EncoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(heads,
                                                 dim_model,
                                                 dim_key,
                                                 dim_value,
                                                 dropout=dropout)
        self.pos_ff = PositionWiseFeedForward(dim_model,
                                              dim_inner,
                                              dropout=dropout)

    def forward(self, encoder_input, self_attention_mask=None):

        encoder_output, encoder_self_attention = self.self_attention(
            encoder_input,
            encoder_input,
            encoder_input,
            mask=self_attention_mask)

        encoder_output = self.pos_ff(encoder_output)

        return encoder_output, encoder_self_attention


class DecoderLayer(nn.Module):
    def __init__(self,
                 dim_model,
                 dim_inner,
                 heads,
                 dim_key,
                 dim_value,
                 dropout=0.1):

        super(DecoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(heads,
                                                 dim_model,
                                                 dim_key,
                                                 dim_value,
                                                 dropout=dropout)
        self.encoder_attention = MultiHeadAttention(heads,
                                                    dim_model,
                                                    dim_key,
                                                    dim_value,
                                                    dropout=dropout)

        self.pos_ff = PositionWiseFeedForward(dim_model,
                                              dim_inner,
                                              dropout=dropout)

    def forward(self,
                decoder_input,
                encoder_output,
                self_attention_mask=None,
                decoder_encoder_attention_mask=None):

        decoder_output, decoder_self_attention = self.self_attention(
            decoder_input,
            decoder_input,
            decoder_input,
            mask=self_attention_mask)

        decoder_output, decoder_encoder_attention = self.encoder_attention(
            decoder_output,
            encoder_output,
            encoder_output,
            mask=decoder_encoder_attention_mask)

        decoder_output = self.pos_ff(decoder_output)

        return decoder_output, decoder_self_attention, decoder_encoder_attention
