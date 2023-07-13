import math
from typing import Optional

import mindspore as ms
import mindspore.nn as nn
from mindspore import ops


class AttentionBlock(nn.Cell):
    """
    An attention block that allows spatial positions to attend to each other. Originally ported from here, but adapted
    to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    Uses three q, k, v linear layers to compute attention.

    Parameters:
        channels (`int`): The number of channels in the input and output.
        num_head_channels (`int`, *optional*):
            The number of channels in each head. If None, then `num_heads` = 1.
        norm_num_groups (`int`, *optional*, defaults to 32): The number of groups to use for group norm.
        rescale_output_factor (`float`, *optional*, defaults to 1.0): The factor to rescale the output by.
        eps (`float`, *optional*, defaults to 1e-5): The epsilon value to use for group norm.
    """

    # IMPORTANT;TODO(Patrick, William) - this class will be deprecated soon. Do not use it anymore

    def __init__(
            self,
            channels: int,
            num_head_channels: Optional[int] = None,
            norm_num_groups: int = 32,
            rescale_output_factor: float = 1.0,
            eps: float = 1e-5,
    ):
        super().__init__()
        self.channels = channels

        self.num_heads = channels // num_head_channels if num_head_channels is not None else 1
        self.num_head_size = num_head_channels
        self.group_norm = nn.GroupNorm(num_channels=channels, num_groups=norm_num_groups, eps=eps, affine=True)

        # define q,k,v as linear layers
        self.query = nn.Dense(channels, channels).to_float(ms.float16)
        self.key = nn.Dense(channels, channels).to_float(ms.float16)
        self.value = nn.Dense(channels, channels).to_float(ms.float16)

        self.rescale_output_factor = rescale_output_factor
        self.proj_attn = nn.Dense(channels, channels, 1).to_float(ms.float16)

        self._use_memory_efficient_attention_xformers = False

    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.num_heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = ops.transpose(tensor, (0, 2, 1, 3))
        tensor = ops.reshape(tensor, (batch_size * head_size, seq_len, dim // head_size))
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.num_heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = ops.transpose(tensor, (0, 2, 1, 3))
        tensor = ops.reshape(tensor, (batch_size // head_size, seq_len, dim * head_size))
        return tensor

    def construct(self, hidden_states):
        residual = hidden_states
        batch, channel, height, width = hidden_states.shape

        # norm
        hidden_states = self.group_norm(hidden_states)

        hidden_states = hidden_states.view(batch, channel, height * width).transpose((0, 2, 1))

        # proj to q, k, v
        query_proj = self.query(hidden_states)
        key_proj = self.key(hidden_states)
        value_proj = self.value(hidden_states)

        scale = 1 / math.sqrt(self.channels / self.num_heads + 1e-5)

        query_proj = self.reshape_heads_to_batch_dim(query_proj)
        key_proj = self.reshape_heads_to_batch_dim(key_proj)
        value_proj = self.reshape_heads_to_batch_dim(value_proj)

        if self._use_memory_efficient_attention_xformers:
            # Memory efficient attention
            hidden_states = hidden_states.to(query_proj.dtype)
        else:
            empty_tensor = ms.numpy.empty((
                query_proj.shape[0],
                query_proj.shape[1],
                key_proj.shape[1]),
                dtype=query_proj.dtype,
            )
            empty_tensor = ms.Tensor(empty_tensor)
            attention_scores = ops.baddbmm(
                empty_tensor,
                query_proj,
                key_proj.transpose((0, 2, 1)),
                beta=0,
                alpha=scale,
            )
            cast = ops.Cast()
            attention_probs = ops.softmax(cast(attention_scores, ms.float16), axis=-1).astype(attention_scores.dtype)
            hidden_states = ops.bmm(attention_probs, value_proj)

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)

        # compute next hidden_states
        hidden_states = self.proj_attn(hidden_states)

        hidden_states = hidden_states.transpose((0, 2, 1)).reshape(batch, channel, height, width)

        # res connect and rescale
        hidden_states = (hidden_states + residual) / self.rescale_output_factor
        return hidden_states


class Attention(nn.Cell):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super(Attention, self).__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Dense(dim, dim * 3, has_bias=qkv_bias).to_float(ms.float16)
        self.attn_drop = nn.Dropout(1.0 - attn_drop)
        self.proj = nn.Dense(dim, dim).to_float(ms.float16)
        self.proj_drop = nn.Dropout(1.0 - proj_drop)

    def construct(self, x):
        unstack = ms.ops.Unstack()
        transpose = ms.ops.Transpose()
        input_perm = (2, 0, 3, 1, 4)
        B, N, C = x.shape
        qkv = transpose(self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads), input_perm)
        q, k, v = unstack(qkv)
        batmatmul = ms.ops.BatchMatMul()
        attn = (batmatmul(q, transpose(k, (0, 1, 3, 2)))) * self.scale
        attn = ms.ops.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = transpose((batmatmul(attn, v)), (0, 2, 1, 3)).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
