import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, dim_hid, num_pos):

        super(PositionalEncoding, self).__init__()

        self.register_buffer(
            'pos_table', self._get_sinusoid_encoding_table(num_pos, dim_hid))

    def _get_sinusoid_encoding_table(self, num_pos, dim_hid):
        def get_position_angle_vec(position):

            return [
                position / np.power(10000, 2 * (hid_j // 2) / dim_hid)
                for hid_j in range(dim_hid)
            ]

        sinusoid_table = np.array(
            [get_position_angle_vec(pos_i) for pos_i in range(num_pos)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class PositionWiseFeedForward(nn.Module):
    '''
    Feed Forward Layer
    '''
    def __init__(self, dim_in, dim_hid, elu_func: str = 'gelu', dropout=0.1):

        super().__init__()
        activation_dict = {
            # rectified linear unit
            'relu': torch.relu,
            # randomized rectified linear unit
            'rrelu': torch.rrelu,
            # relu 6 pad = 6
            'relu6': nn.ReLU6(),
            # Exponential linear unit
            'elu': nn.ELU(),
            # continuously differentiable exponential linear units
            'celu': nn.CELU(),
            # self-normalizing exponential linear units
            'selu': nn.SELU(),
            # gaussian error linear units
            'gelu': F.gelu,
            # parametric rectified linear units
            'prelu': nn.PReLU()
        }
        self.feedforward_1 = nn.Linear(dim_in, dim_hid)
        self.feedforward_2 = nn.Linear(dim_hid, dim_in)

        self.layer_norm = nn.LayerNorm(dim_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.elu = activation_dict[elu_func]

    def forward(self, inp):

        residual = inp
        inp = self.layer_norm(inp)

        inp = self.feedforward_2(self.elu(self.feedforward_1(inp)))

        inp = self.dropout(inp)

        inp += residual

        return inp