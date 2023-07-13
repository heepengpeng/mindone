""" MLP module w/ dropout and configurable activation layer

Hacked together by / Copyright 2020 Ross Wightman
"""

import mindspore as ms
import mindspore.nn as nn
from utils.util import _ntuple

to_2tuple = _ntuple(2)


class Mlp(nn.Cell):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Dense(in_features, hidden_features, has_bias=bias[0]).to_float(ms.float16)
        self.act = act_layer()
        self.drop1 = nn.Dropout(1.0 - float(drop_probs[0]))
        self.fc2 = nn.Dense(hidden_features, out_features, has_bias=bias[1]).to_float(ms.float16)
        self.drop2 = nn.Dropout(1.0 - drop_probs[1])

    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
