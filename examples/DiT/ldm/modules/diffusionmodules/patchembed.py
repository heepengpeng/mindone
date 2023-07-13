""" Image to Patch Embedding using Conv2d

A convolution based approach to patchifying a 2D image w/ embedding projection.

Based on the impl in https://github.com/google-research/vision_transformer

Hacked together by / Copyright 2020 Ross Wightman
"""
from mindspore import nn as nn
from mindspore import ops
from itertools import repeat
import collections.abc
import mindspore as ms


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


class PatchEmbed(nn.Cell):
    """ 2D Image to Patch Embedding
    """

    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super(PatchEmbed, self).__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, pad_mode='valid',
                              has_bias=bias
                              ).to_float(ms.float16)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def construct(self, x):
        B, C, H, W = x.shape
        assert (H == self.img_size[0])
        assert (W == self.img_size[1])
        x = self.proj(x)
        if self.flatten:
            # BCHW -> BNC
            b, c, h, w = x.shape
            x = ops.reshape(x, (b, c, h * w))
            x = ops.transpose(x, (0, 2, 1))
        x = self.norm(x)
        return x
