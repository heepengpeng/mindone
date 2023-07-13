import json
from typing import Tuple

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as P
from mindspore import save_checkpoint

from ldm.modules.diffusionmodules.resnet import UNetMidBlock2D
from ldm.modules.diffusionmodules.unet_2d_blocks import get_down_block, get_up_block
import numpy as np


class AutoencoderKL(nn.Cell):

    def __init__(self, ckpt_path, config_path, is_torch_model=True,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
                 up_block_types: Tuple[str] = ("UpDecoderBlock2D",),
                 block_out_channels: Tuple[int] = (4,),
                 layers_per_block: int = 2,
                 act_fn: str = "silu",
                 latent_channels: int = 4,
                 norm_num_groups: int = 32):
        super().__init__()
        if config_path:
            with open(config_path) as config_file:
                config_json = json.load(config_file)
                in_channels = config_json["in_channels"]
                out_channels = config_json["out_channels"]
                down_block_types = config_json["down_block_types"]
                up_block_types = config_json["up_block_types"]
                block_out_channels = config_json["block_out_channels"]
                layers_per_block = config_json["layers_per_block"]
                act_fn = config_json["act_fn"]
                latent_channels = config_json["latent_channels"]
                norm_num_groups = config_json["norm_num_groups"]
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
        )

        # pass init params to Decoder
        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
        )
        self.is_torch_model = is_torch_model
        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1, has_bias=True)
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1, has_bias=True)
        self.use_slicing = False
        self.split = P.Split(axis=1, output_num=2)
        if ckpt_path:
            self.init_from_ckpt(ckpt_path)

    def init_from_ckpt(self, path):
        if self.is_torch_model:
            import torch
            state_dict = torch.load(path, map_location=torch.device('cpu'))
            ms_ckpt = torch_to_ms(state_dict)
            path = path.replace('.bin', '.ckpt')
            save_checkpoint(ms_ckpt, path)
        sd = ms.load_checkpoint(path)
        ms.load_param_into_net(self, sd, strict_load=False)

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        mean, logvar = self.split(moments)
        logvar = P.clip_by_value(logvar, -30.0, 20.0)
        std = P.exp(0.5 * logvar)
        x = mean + std * ms.Tensor.from_numpy(np.random.standard_normal(mean.shape)).astype(ms.float16)
        return x

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec


def torch_to_ms(state_dict):
    ms_ckpt = []
    for k, v in state_dict.items():
        if 'norm' in k:
            k = k.replace('weight', 'gamma')
            k = k.replace('bias', 'beta')
        ms_ckpt.append({'name': k, 'data': ms.Tensor(v.numpy())})
    return ms_ckpt


class Encoder(nn.Cell):
    def __init__(
            self,
            in_channels=3,
            out_channels=3,
            down_block_types=("DownEncoderBlock2D",),
            block_out_channels=(64,),
            layers_per_block=2,
            norm_num_groups=32,
            act_fn="silu",
            double_z=True,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1, pad_mode="pad",
                                 has_bias=True)

        self.mid_block = None
        self.down_blocks = nn.CellList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                resnet_eps=1e-6,
                downsample_padding=0,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attn_num_head_channels=None,
                temb_channels=None,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attn_num_head_channels=None,
            resnet_groups=norm_num_groups,
            temb_channels=None,
        )

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = nn.Conv2d(block_out_channels[-1], conv_out_channels, 3, padding=1, pad_mode="pad",
                                  has_bias=True)

    def construct(self, x):
        sample = x
        sample = self.conv_in(sample)

        # down
        for down_block in self.down_blocks:
            sample = down_block(sample)

        # middle
        sample = self.mid_block(sample)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample


class Decoder(nn.Cell):
    def __init__(
            self,
            in_channels=3,
            out_channels=3,
            up_block_types=("UpDecoderBlock2D",),
            block_out_channels=(64,),
            layers_per_block=2,
            norm_num_groups=32,
            act_fn="silu",
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2d(in_channels, block_out_channels[-1], kernel_size=3, stride=1, padding=1,
                                 pad_mode="pad", has_bias=True)

        self.mid_block = None
        self.up_blocks = nn.CellList([])

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attn_num_head_channels=None,
            resnet_groups=norm_num_groups,
            temb_channels=None,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                prev_output_channel=None,
                add_upsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attn_num_head_channels=None,
                temb_channels=None,
            )
            self.up_blocks.append(up_block)

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1, pad_mode="pad", has_bias=True)

    def construct(self, z):
        sample = z
        sample = self.conv_in(sample)
        # middle
        sample = self.mid_block(sample)
        # up
        for up_block in self.up_blocks:
            sample = up_block(sample)
        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        return sample
