import argparse
import logging
import os
import random
import string
import time

import mindspore as ms
import numpy as np
from mindspore import ops

from ldm.models.diffusion import create_diffusion
from ldm.modules.diffusionmodules.models import DiT_models
from ldm.modules.encoders.vae import AutoencoderKL
from utils.model_utils import load_model, load_ckpt
from utils.util import save_image

logging.basicConfig(
    level=logging.INFO,
    format='[\033[34m%(asctime)s\033[0m] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler()])


# util
def mkdir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def main(args):
    if args.profile_path:
        profile = ms.Profiler(output_path=args.profile_path)
    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    num_sampling_steps = args.num_sampling_steps
    cfg_scale = args.cfg_scale
    # Load model:
    image_size = args.image_size
    assert image_size in [256, 512], "We only provide pre-trained models for 256x256 and 512x512 resolutions."
    latent_size = image_size // 8
    model = DiT_models[args.model](input_size=latent_size, num_classes=args.num_classes)
    if args.model_path:
        state_dict = load_ckpt(args.model_path)
    else:
        state_dict = load_model(f"DiT-XL-2-{image_size}x{image_size}.pt")
    ms.load_param_into_net(model, state_dict)
    model.set_train(False)

    # vae model
    vae = AutoencoderKL(args.vae_ckpt_path, args.vae_config_path)
    batch_size = args.batch_size
    generate_num = 0
    while generate_num < args.num_samples:
        random_int_op = ops.UniformInt(seed=int(time.time()))
        if args.random:
            class_labels = random_int_op((args.batch_size,), ms.Tensor(0, ms.int32),
                                         ms.Tensor(args.num_classes, ms.int32))
            class_labels = class_labels.astype(ms.int64)
        else:
            class_labels = ms.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], dtype=ms.int64)
        diffusion = create_diffusion(str(num_sampling_steps))
        n = len(class_labels)
        shape = (n, 4, latent_size, latent_size)
        z = ms.Tensor.from_numpy(np.random.standard_normal(shape)).astype(ms.float16)
        concat_op = ms.ops.Concat(axis=0)
        z = concat_op((z, z))
        y = ms.Tensor(class_labels)
        # Setup classifier-free guidance:
        concat_op = ms.ops.Concat(axis=0)
        y_null = ms.Tensor([1000] * n)
        y = concat_op((y, y_null))
        z = z.astype(ms.float16)
        # Sample images:
        if cfg_scale:
            model_kwargs = dict(y=y, cfg_scale=cfg_scale)
            samples = diffusion.p_sample_loop(
                model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True
            )
        else:
            model_kwargs = dict(y=y)
            samples = diffusion.p_sample_loop(
                model, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True
            )
        split = ms.ops.Split(axis=0, output_num=2)
        samples, _ = split(samples)
        # Remove null class samples
        samples = vae.decode(samples.astype(ms.float32) / 0.18215)
        save_dir = "sample/image/"
        mkdir_if_not_exist(save_dir)
        for sample in samples:
            save_image(sample, save_dir + "/sample%s.png" % (''.join(random.sample(string.ascii_letters, 8))),
                       nrow=1,
                       normalize=True,
                       value_range=(-1, 1))
        generate_num = generate_num + batch_size
        logging.info("generated_num %d", generate_num)
    if args.profile_path:
        profile.analyse()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=6400)
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--image_size', default=256, type=int)
    parser.add_argument('--out_dir', default='./')
    parser.add_argument('--num_sampling_steps', default=250, type=int)
    parser.add_argument('--cfg_scale', default=None, type=float)
    parser.add_argument('--profile_path', default=None, type=str)
    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument('--random', default=True, type=bool)
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae_ckpt_path", type=str, required=True, help="path to vae checkpoint")
    parser.add_argument("--vae_config_path", type=str, required=True, help="path to vae config path")
    args = parser.parse_args()
    print(args)
    main(args)
