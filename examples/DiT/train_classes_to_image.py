import argparse
import logging
import os
import sys

import mindspore as ms

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../")))
from mindspore import ops, context, Model, TimeMonitor, LossMonitor
from mindspore.amp import DynamicLossScaler
from mindspore.communication import init, get_rank, get_group_size
from mindspore.nn.optim.adam import AdamWeightDecay

from ldm.data.dataset import build_dataset
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.modules.diffusionmodules.models import DiT_models
from ldm.modules.logger import set_logger
from stable_diffusion_v2.ldm.modules.train.callback import OverflowMonitor, EvalSaveCallback
from stable_diffusion_v2.ldm.modules.train.parallel_config import ParallelConfig
from stable_diffusion_v2.ldm.modules.train.trainer import TrainOneStepWrapper
from stable_diffusion_v2.ldm.modules.train.tools import set_random_seed
from stable_diffusion_v2.ldm.util import str2bool

os.environ["HCCL_CONNECT_TIMEOUT"] = "6000"

logger = logging.getLogger(__name__)


def init_env(args):
    set_random_seed(args.seed)

    ms.set_context(mode=context.GRAPH_MODE)  # needed for MS2.0
    if args.use_parallel:
        init()
        device_id = int(os.getenv("DEVICE_ID"))
        device_num = get_group_size()
        ParallelConfig.dp = device_num
        rank_id = get_rank()
        args.rank = rank_id
        logger.debug("Device_id: {}, rank_id: {}, device_num: {}".format(device_id, rank_id, device_num))
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(
            parallel_mode=context.ParallelMode.DATA_PARALLEL,
            # parallel_mode=context.ParallelMode.AUTO_PARALLEL,
            gradients_mean=True,
            device_num=device_num,
        )
    else:
        device_num = 1
        device_id = int(os.getenv("DEVICE_ID", 0))
        rank_id = 0
        args.rank = rank_id

    context.set_context(
        mode=context.GRAPH_MODE,
        device_target="Ascend",
        device_id=device_id,
        max_device_memory="30GB",  # TODO: why limit?
    )
    return rank_id, device_id, device_num


def main(args):
    # init
    rank_id, device_id, device_num = init_env(args)
    set_logger(name="", output_dir=args.output_path, rank=rank_id, log_level=eval(args.log_level))

    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."

    # build dataset
    dataset = build_dataset(args=args, device_num=device_num, rank_id=rank_id)
    logger.info(f"Dataset size: {dataset.get_dataset_size()}")

    # build model
    latent_size = args.image_size // 8
    dit_model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    )
    latent_diffusion_with_loss = LatentDiffusion(model=dit_model, args=args)

    logger.info(f"DiT Parameters: {sum(ops.size(p) for p in dit_model.get_parameters()):,}")

    optimizer = AdamWeightDecay(params=latent_diffusion_with_loss.model.trainable_params(), learning_rate=args.lr,
                                weight_decay=0)

    loss_scaler = DynamicLossScaler(args.loss_scale, args.scale_factor, args.scale_window)

    net_with_grads = TrainOneStepWrapper(
        latent_diffusion_with_loss,
        optimizer=optimizer,
        scale_sense=loss_scaler,
        drop_overflow_update=True,  # TODO: allow config
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        clip_grad=args.clip_grad,
        clip_norm=args.max_grad_norm,
        ema=None,
    )

    model = Model(net_with_grads)

    # callbacks
    callback = [TimeMonitor(args.callback_size), LossMonitor(args.callback_size)]
    ofm_cb = OverflowMonitor()
    callback.append(ofm_cb)

    if rank_id == 0:
        ckpt_dir = os.path.join(args.output_path, "ckpt", f"rank_{str(rank_id)}")
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        save_cb = EvalSaveCallback(
            network=dit_model,
            use_lora=args.use_lora,
            rank_id=rank_id,
            ckpt_save_dir=ckpt_dir,
            ema=None,
            ckpt_save_policy="latest_k",
            ckpt_max_keep=10,
            ckpt_save_interval=args.ckpt_save_interval,
            lora_rank=args.lora_rank,
        )
        callback.append(save_cb)

        # log
        if rank_id == 0:
            key_info = "Key Settings:\n" + "=" * 50 + "\n"
            key_info += "\n".join(
                [
                    "MindSpore mode[GRAPH(0)/PYNATIVE(1)]: 0",
                    f"Distributed mode: {args.use_parallel}",
                    f"Data path: {args.data_path}",
                    f"Precision: {dit_model.dtype}",
                    f"Learning rate: {args.start_learning_rate}",
                    f"Batch size: {args.train_batch_size}",
                    f"Num epochs: {args.epochs}",
                ]
            )
            key_info += "\n" + "=" * 50
            logger.info(key_info)
            logger.info("Start training...")
        # train
        model.train(args.epochs, dataset, callbacks=callback, dataset_sink_mode=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_parallel", default=False, type=str2bool, help="use parallel")
    parser.add_argument("--data_path", required=True, type=str, help="data path")
    parser.add_argument("--output_path", default="output/", type=str, help="output directory to save training results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae_ckpt_path", type=str, required=True, help="path to vae checkpoint")
    parser.add_argument("--vae_config_path", type=str, required=True, help="path to vae config path")
    parser.add_argument("--image_size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global_batch_size", type=int, default=128)
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--ckpt_every", type=int, default=50_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--amp_level", type=str, default="O2")
    parser.add_argument("--loss_scale", type=int, default=1024)
    parser.add_argument("--pretrained", type=bool, default=False)
    parser.add_argument("--scale_factor", type=int, default=2)
    parser.add_argument("--scale_window", type=int, default=50)
    parser.add_argument(
        "--log_level",
        type=str,
        default="logging.INFO",
        help="log level, options: logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR",
    )
    parser.add_argument("--clip_grad", default=False, type=str2bool, help="whether apply gradient clipping")
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="max gradient norm for clipping, effective when `clip_grad` enabled.",
    )
    args = parser.parse_args()
    print(args)
    main(args)
