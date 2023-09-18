import argparse
import logging
import os
import sys
import time

import mindspore as ms
from mindspore import ops
from omegaconf import OmegaConf

workspace = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(workspace))
from libs.logger import set_logger
from libs.sd_models import SDText2Img
from libs.helper import set_env, load_model_from_config, VaeImageProcessor
from gm.util import instantiate_from_config
from libs.util import str2bool

logger = logging.getLogger("Stable Diffusion XL Inference")


def main(args):
    # set logger
    set_env(args)
    ms.set_context(device_target=args.device_target)

    # create model
    config = OmegaConf.load(f"{args.model}")
    model = load_model_from_config(
        config,
        ckpt=config.model.pretrained_ckpt,
        freeze=True, load_filter=False, amp_level=args.ms_amp_level
    )
    sampler_config = OmegaConf.load(args.sampler)
    scheduler = instantiate_from_config(sampler_config)
    timesteps = scheduler.set_timesteps(args.sampling_steps)

    # read prompts
    batch_size = args.n_samples
    prompt = args.inputs.prompt
    data = batch_size * [prompt]
    negative_prompt = args.inputs.negative_prompt
    assert negative_prompt is not None
    negative_data = batch_size * [negative_prompt]

    # post-process negative prompts
    assert len(negative_data) <= len(data), "Negative prompts should be shorter than positive prompts"
    if len(negative_data) < len(data):
        logger.info("Negative prompts are shorter than positive prompts, padding blank prompts")
        blank_negative_prompt = batch_size * [""]
        for _ in range(len(data) - len(negative_data)):
            negative_data.append(blank_negative_prompt)

    # create inputs
    inputs = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "orig_width": args.orig_width if args.orig_width else W,
        "orig_height": args.orig_height if args.orig_height else H,
        "target_width": args.target_width if args.target_width else W,
        "target_height": args.target_height if args.target_height else H,
        "crop_coords_top": max(args.crop_coords_top if args.crop_coords_top else 0, 0),
        "crop_coords_left": max(args.crop_coords_left if args.crop_coords_left else 0, 0),
        "aesthetic_score": 6.0,
        "negative_aesthetic_score": 2.5,
        "timesteps": timesteps,
        "scale": ms.Tensor(args.scale, ms.float16),
        "num_samples": 1,
        "dtype": ms.float16
    }

    # create model
    text_encoder = model.conditioner
    unet = model.model
    vae = model.first_stage_model
    img_processor = VaeImageProcessor()

    if args.device_target != "Ascend":
        unet.to_float(ms.float32)
        vae.to_float(ms.float32)

    if args.task == "text2img":
        sd_infer = SDText2Img(
            text_encoder,
            unet,
            vae,
            scheduler,
            scale_factor=model.scale_factor,
            num_inference_steps=args.sampling_steps,
        )
    else:
        raise ValueError(f"Not support task: {args.task}")

    logger.info(
        f"Generating images with conditions:\n"
        f"Prompt(s): {inputs['prompt']}, \n"
        f"Negative prompt(s): {inputs['negative_prompt']}"
    )

    for n in range(args.n_iter):
        start_time = time.time()
        inputs["noise"] = ops.standard_normal((args.n_samples, 4, args.inputs.H // 8, args.inputs.W // 8)).astype(
            ms.float16
        )
        x_samples = sd_infer(inputs)
        x_samples = img_processor.postprocess(x_samples)

        for sample in x_samples:
            sample.save(os.path.join(args.sample_path, f"{args.base_count:05}.png"))
            args.base_count += 1

        end_time = time.time()
        logger.info(
            "{}/{} images generated, time cost for current trial: {:.3f}s".format(
                batch_size * (n + 1), batch_size * args.n_iter, end_time - start_time
            )
        )

    logger.info(f"Done! All generated images are saved in: {args.output_path}/samples" f"\nEnjoy.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ms_mode", type=int, default=0, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)"
    )
    parser.add_argument(
        "--device_target", type=str, default="Ascend", help="Device target, should be in [Ascend, GPU, CPU]"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="text2img",
        help="Task name, should be [text2img, img2img, inpaint], "
             "if choose a task name, use the config/[task].yaml for inputs",
        choices=["text2img", "img2img", "inpaint"],
    )
    parser.add_argument("--model", type=str, required=True, help="path to config which constructs model.")
    parser.add_argument("--output_path", type=str, default="output", help="dir to write results to")
    parser.add_argument("--sampler", type=str, default="./config/schedule/dpmsolver_multistep.yaml",
                        help="infer sampler yaml path")
    parser.add_argument("--sampling_steps", type=int, default=50, help="number of sampling steps")
    parser.add_argument("--n_iter", type=int, default=1, help="number of iterations or trials.")
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt in an iteration. A.k.a. batch size",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=9.0,
        help="unconditional guidance scale: "
             "eps = eps(x, uncond) + scale * (eps(x, cond) - eps(x, uncond)). "
             "Simplified: `uc + scale * (uc - prompt)`",
    )
    parser.add_argument(
        "--use_lora",
        default=False,
        type=str2bool,
        help="whether the checkpoint used for inference is finetuned from LoRA",
    )
    parser.add_argument(
        "--lora_rank",
        default=None,
        type=int,
        help="LoRA rank. If None, lora checkpoint should contain the value for lora rank in its append_dict.",
    )
    parser.add_argument(
        "--lora_ckpt_path", type=str, default=None, help="path to lora only checkpoint. Set it if use_lora is not None"
    )
    parser.add_argument("--seed", type=int, default=42, help="the seed (for reproducible sampling)")
    parser.add_argument("--log_level", type=str, default="INFO", help="log level, options: DEBUG, INFO, WARNING, ERROR")
    args = parser.parse_args()
    set_logger(name="", output_dir=args.output_path, rank=0, log_level=args.log_level)

    if not os.path.exists(args.model):
        raise ValueError(
            f"model config file {args.model} is not exist!, please set it by --model=xxx.yaml. "
            f"eg. --model=./config/model/v2-inference.yaml"
        )
    if not os.path.isabs(args.model):
        args.model = os.path.join(workspace, args.model)
    if args.task == "text2img":
        inputs_config_path = "./config/text2img.yaml"
        default_ckpt = "./models/sd_v2_base-57526ee4.ckpt"
    else:
        raise ValueError(f"{args.task} is invalid, should be in [text2img, img2img, inpaint]")
    inputs = OmegaConf.load(inputs_config_path)

    key_settings_info = ["Key Settings:\n" + "=" * 50]
    key_settings_info += [
        f"SD infer task: {args.task}",
        f"model config: {args.model}",
        f"inputs config: {inputs_config_path}",
        f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.ms_mode}",
        f"Number of trials for each prompt: {args.n_iter}",
        f"Number of samples in each trial: {args.n_samples}",
        f"Sampler: {args.sampler}",
        f"Sampling steps: {args.sampling_steps}",
        f"Uncondition guidance scale: {args.scale}",
    ]
    for key in inputs.keys():
        key_settings_info.append(f"{key}: {inputs[key]}")

    logger.info("\n".join(key_settings_info))

    args.inputs = inputs
    main(args)
