import logging

import diffusers
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils import data
import torchvision

import configs
import utils

_CFG = {
    "experiment": utils.get_current_ts(),
    "data": {
        "n_classes": 10,
    },
    "train": {
        "unet": {
            "sample_size": 32,
            "out_channels": 3,
        }
    },
    "diffusion": {
        "max_time_steps": 1000,
        "beta_schedule": "squaredcos_cap_v2",
    },
    "path": {
        "model":
            "runs_cc/20240906T212532M370/checkpoints/model_74",
        "gen_file":
            "runs_cc/20240906T212532M370/generated_images/{experiment}/plots/{epoch}.png"
    }
}


class ClassConditionedUnet(nn.Module):

  def __init__(self, cfg, class_emb_size=4):
    super().__init__()

    # The embedding layer will map the class label to a vector of size class_emb_size
    self.class_emb = nn.Embedding(cfg["data", "n_classes"], class_emb_size)

    # Self.model is an unconditional UNet with extra input channels to accept the conditioning information (the class embedding)
    out_channels = cfg["train", "unet", "out_channels"]
    self.model = diffusers.UNet2DModel(
        sample_size=cfg["train", "unet", "sample_size"],
        in_channels=out_channels + class_emb_size,
        out_channels=cfg["train", "unet", "out_channels"],
        layers_per_block=2,
        block_out_channels=(32, 64, 64),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
    )

  def forward(self, x, t, class_labels):
    bs, ch, w, h = x.shape

    # class conditioning in right shape to add as additional input channels
    class_cond = self.class_emb(class_labels)
    class_cond = class_cond.view(bs, class_cond.shape[1], 1,
                                 1).expand(bs, class_cond.shape[1], w, h)
    # x is shape (bs, 1, 28, 28) and class_cond is now (bs, 4, 28, 28)

    # Net input is now x and class cond concatenated together along dimension 1
    net_input = torch.cat((x, class_cond), 1)  # (bs, 5, 28, 28)

    # Feed this to the UNet alongside the timestep and return the prediction
    return self.model(net_input, t).sample  # (bs, 1, 28, 28)


def get_noise_scheduler(cfg):
  return diffusers.DDPMScheduler(
      num_train_timesteps=cfg["diffusion", "max_time_steps"],
      beta_schedule=cfg["diffusion", "beta_schedule"],
  )


def get_device():
  if torch.backends.mps.is_available():
    device = "mps"
  elif torch.cuda.is_available():
    device = "cuda"
  else:
    device = "cpu"
  return device


def infer(noise_scheduler, net, cfg, f_name):
  device = get_device()

  out_channels = cfg["train", "unet", "out_channels"]
  sample_size = cfg["train", "unet", "sample_size"]
  x = torch.randn(80, out_channels, sample_size, sample_size).to(device)
  y = torch.tensor([[i] * 8 for i in range(10)]).flatten().to(device)

  time_steps = noise_scheduler.timesteps
  p_bar = utils.get_p_bar(len(time_steps))
  for idx, t in enumerate(time_steps):
    with torch.no_grad():
      residual = net(x, t, y)
    x = noise_scheduler.step(residual, t, x).prev_sample
    p_bar.update(1)

    if cfg["args", "debug"] and idx == 20:
      break

  p_bar.close()
  _, ax = plt.subplots(1, 1, figsize=(12, 12))

  grid = torchvision.utils.make_grid(x.detach().cpu().clip(-1, 1), nrow=8)
  grid = grid.permute(1, 2, 0)
  grid = grid[..., [2, 1, 0]]
  ax.imshow((grid + 1) / 2)

  gen_file = utils.get_path(cfg, "gen_file", epoch=f_name)
  plt.savefig(gen_file)


def main():
  args = utils.parse_args()
  if args.debug:
    _CFG["experiment"] = f"{_CFG['experiment']}_debug"

  cfg = configs.Configs(_CFG, args.configs, args, False)
  logging.info(f"Experiment: {cfg['experiment']}")

  device = get_device()

  net = ClassConditionedUnet(cfg).to(device)
  noise_scheduler = get_noise_scheduler(cfg)

  model_path = utils.get_path(cfg, "model")
  net.load_state_dict(torch.load(model_path), weights_only=True)

  infer(noise_scheduler, net, cfg, "pred")


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  main()
