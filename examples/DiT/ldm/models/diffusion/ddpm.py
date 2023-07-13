import mindspore as ms
from mindspore import nn
from mindspore import ops

from ldm.models.diffusion import create_diffusion
from ldm.modules.encoders.vae import AutoencoderKL


class LatentDiffusion(nn.Cell):

    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.diffusion = create_diffusion("")
        self.vae = AutoencoderKL(ckpt_path=args.vae_ckpt_path, config_path=args.vae_config_path,
                                 is_torch_model=True)

    def construct(self, x, y):
        x = ops.mul(self.vae.encode(x), 0.18215).astype(ms.float16)
        t = ms.numpy.randint(0, self.diffusion.num_timesteps, (x.shape[0],))
        model_kwargs = dict(y=y)
        loss = self.diffusion.training_losses(self.dit_model, x, t, model_kwargs)
        return ms.ops.mean(loss).astype(ms.float32)
