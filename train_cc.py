import logging
import os
from typing import List, Tuple

import torch
from torch import nn
from torch.utils import data
import torchvision

import configs
import infer_cc
import utils
import utils_cc

_CFG = {
    "experiment": utils.get_current_ts(),
    "data": {
        "set": "mnist",
        "n_classes": 10,
        "retrain_classes": None,
    },
    "train": {
        "batch_size": 128,
        "epochs": 100,
        "lr": 1e-3,
        "save_at": [5, 10, 25, 50, 75, 100],
        "unet": {
            "sample_size": 28,
            "out_channels": 1,
            "layers_per_block": 2,
            "block_out_channels": (32, 64, 64),
            "down_block_types":
                ("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
            "up_block_types": ("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        },
        "ascent": False,
    },
    "diffusion": {
        "max_time_steps": 1000,
        "beta_schedule": "squaredcos_cap_v2",
        "infer_at": [1, 5, 10, 25, 50, 75, 100],
    },
    "path": {
        "prefix":
            "",
        "model":
            "runs_cc/{experiment}/checkpoints/model_{epoch}",
        "gen_file":
            "runs_cc/{experiment}/generated_images/{experiment}/plots/{epoch}.png",
        "configs":
            "runs_cc/{experiment}/configs.json",
        "checkpoint":
            "",
    },
    "infer_cfg": {
        "n_images_per_class": 1000,
        "format": ["collage"],
        "num_inference_steps": 64
    }
}

_TYPE = {
    "experiment": str,
    "data": {
        "set": str,
        "n_classes": int,
        "retrain_classes": List[int],
    },
    "train": {
        "batch_size": int,
        "epochs": int,
        "lr": float,
        "save_at": List[int],
        "unet": {
            "sample_size": int,
            "out_channels": int,
            "layers_per_block": int,
            "block_out_channels": Tuple[int, ...],
            "down_block_types": Tuple[str, ...],
            "up_block_types": Tuple[str, ...],
        },
        "ascent": bool,
    },
    "diffusion": {
        "max_time_steps": int,
        "beta_schedule": str,
        "infer_at": List[int],
    },
    "path": {
        "model": str,
        "gen_file": str,
        "configs": str,
        "ind_path": str,
        "checkpoint": str,
        "test_images": str,
        "img_lab_path": str,
    },
    "infer_cfg": {
        "n_images_per_class": int,
        "format": List[str],
        "num_inference_steps": int
    }
}


def get_data(cfg):
  data_name = cfg["data", "set"]
  match (data_name):
    case "mnist":
      tv_cl = torchvision.datasets.MNIST
    case "cifar10":
      tv_cl = torchvision.datasets.CIFAR10
    case _:
      raise ValueError("Invalid dataset")
  dataset = tv_cl(
      root=data_name,
      train=True,
      download=True,
      transform=torchvision.transforms.ToTensor(),
  )
  retrain_cls = cfg["data", "retrain_classes"]
  if retrain_cls:
    indices = [idx for idx, (_, lb) in enumerate(dataset) if lb in retrain_cls]
    dataset = torch.utils.data.Subset(dataset, indices)
  train_dataloader = data.DataLoader(
      dataset,
      batch_size=cfg["train", "batch_size"],
      shuffle=True,
  )
  return train_dataloader


class AscentFunction(torch.autograd.Function):

  @staticmethod
  def forward(ctx, input):
    return input

  @staticmethod
  def backward(ctx, grad_input):
    return -grad_input


def make_ascent(loss):
  return AscentFunction.apply(loss)


def main():
  args, unknown_args = utils.parse_args()
  if args.debug:
    _CFG["experiment"] = f"{_CFG['experiment']}_debug"

  cfg = configs.Configs(
      default_configs=_CFG,
      configs_path=args.configs,
      args=args,
      config_args=unknown_args,
      type_hints=_TYPE,
  )
  logging.info(f"Experiment: {cfg['experiment']}")

  device = utils_cc.get_device()

  net = utils_cc.ClassConditionedUnet(cfg).to(device)
  n_parameters = sum(p.numel() for p in net.parameters())
  logging.info(f"Number of parameters: {n_parameters}")

  checkpoint_path = utils.get_path(cfg, "checkpoint")
  if checkpoint_path:
    net.load_state_dict(torch.load(checkpoint_path))

  train_dataloader = get_data(cfg)
  loss_fn = nn.MSELoss()
  opt = torch.optim.Adam(net.parameters(), lr=cfg["train", "lr"])
  noise_scheduler = utils_cc.get_noise_scheduler(cfg)

  debug = cfg["args", "debug"]
  max_time_steps = cfg["diffusion", "max_time_steps"]
  n_epochs = cfg["train", "epochs"]
  save_at = cfg["train", "save_at"]
  infer_at = cfg["diffusion", "infer_at"]
  losses = []

  p_bar = utils.get_p_bar(len(train_dataloader))
  for epoch in range(n_epochs):
    for idx, (x, y) in enumerate(train_dataloader):
      x = x.to(device) * 2 - 1
      y = y.to(device)

      noise = torch.randn_like(x)
      timesteps = torch.randint(0, max_time_steps - 1,
                                (x.shape[0],)).long().to(device)
      noisy_x = noise_scheduler.add_noise(x, noise, timesteps)

      pred = net(noisy_x, timesteps, y)
      loss = loss_fn(pred, noise)

      opt.zero_grad()
      if cfg["train", "ascent"]:
        make_ascent(loss)
      loss.backward()
      opt.step()

      losses.append(loss.item())
      p_bar.update(1)

      if debug and idx > 10:
        break

    if epoch + 1 in save_at or debug:
      model_path = utils.get_path(cfg, "model", epoch=epoch + 1)
      torch.save(net.state_dict(), model_path)
    if epoch + 1 in infer_at or debug:
      logging.info(f"Inferring @ {epoch + 1}")
      infer_cc.infer(cfg, epoch, cfg["infer_cfg", "format"], net)

    avg_loss = sum(losses) / len(losses)
    logging.info(f"Finished epoch {epoch + 1}. Average loss: {avg_loss:05f}")

    if debug and epoch == 0:
      break
  p_bar.close()


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  main()
