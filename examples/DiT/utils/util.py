import collections.abc
import math
import random
from itertools import repeat

import mindspore as ms
import numpy as np
from PIL import Image
from mindspore import ops
from mindspore.dataset.transforms.transforms import PyTensorOperation


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


class Lambda(PyTensorOperation):
    def __init__(self, lambd):
        super().__init__()
        if not callable(lambd):
            raise TypeError(f"Argument lambd should be callable, got{repr(type(lambd).__name__)}")
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return f"{self.__class__.__name__}()"


def save_image(samples, fp, format=None, **kwargs):
    grid = make_grid(samples, **kwargs)
    input_perm = (1, 2, 0)
    transpose = ops.Transpose()
    ndarr = transpose(ops.clip_by_value(ops.add(ops.mul(grid, 255), 0.5), 0, 255), input_perm).astype(
        ms.uint8).asnumpy()
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)


def str2bool(b):
    if b.lower() not in ["false", "true"]:
        raise Exception("Invalid Bool Value")
    if b.lower() in ["false"]:
        return False
    return True


def make_grid(
        tensor,
        nrow=8,
        padding=2,
        normalize=False,
        value_range=None,
        scale_each=False,
        pad_value=0.0,
        **kwargs, ):
    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = ops.Stack(tensor, axis=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = ops.expand_dims(tensor, 0)
    if tensor.dim() == 3:  # single image
        if tensor.shape[0] == 1:  # if single-channel, convert to 3-channel
            tensor = ops.concat((tensor, tensor, tensor), 0)
        tensor = ops.expand_dims(tensor, 0)
    if tensor.dim() == 4 and tensor.shape[1] == 1:  # single-channel images
        tensor = ops.concat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.copy()  # avoid modifying tensor in-place
        if value_range is not None and not isinstance(value_range, tuple):
            raise TypeError("value_range has to be a tuple (min, max) if specified. min and max are numbers")

        def norm_ip(img, low, high):
            img = ops.clip_by_value(img, low, high)
            img = img.sub(low)
            img = ops.div(img, max(high - low, 1e-5))
            return img

        def norm_range(t, value_range):
            if value_range is not None:
                return norm_ip(t, value_range[0], value_range[1])
            else:
                return norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                t = norm_range(t, value_range)
        else:
            tensor = norm_range(tensor, value_range)

    if not isinstance(tensor, ms.Tensor):
        raise TypeError("tensor should be of type torch.Tensor")
    if tensor.shape[0] == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[2] + padding), int(tensor.shape[3] + padding)
    num_channels = tensor.shape[1]
    grid = ms.numpy.full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid[:, y * height + padding:height, :][:, :, x * width + padding:x * width + width] = tensor[
                k].copy()
            k = k + 1
    return grid





def set_random_seed(seed):
    """Set Random Seed"""
    random.seed(seed)
    np.random.seed(seed)
    ms.set_seed(seed)
