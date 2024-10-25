import logging
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.utils import data
import torchvision

import configs
import utils
import utils_cc

_CFG = {
    "experiment": utils.get_current_ts(),
    "data": {
        "n_classes": 10,
    },
    "train": {
        "batch_size": 128,
        "unet": {
            "sample_size": 28,
            "out_channels": 1,
            "layers_per_block": 2,
            "block_out_channels": (32, 64, 64),
            "down_block_types":
                ("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
            "up_block_types": ("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        }
    },
    "diffusion": {
        "max_time_steps": 1000,
        "beta_schedule": "squaredcos_cap_v2",
    },
    "path": {
        "prefix": "runs_cc/20240906T234605M545",
        "model": "checkpoints/model_99",
        "gen_file": "generated_images/{experiment}/plots/{epoch}.png",
        "ind_path": "generated_images/{experiment}/ind/images/{img_id}.png",
        "img_lab_path": "generated_images/{experiment}/ind/img_lab.json",
        "configs": "generated_images/{experiment}/configs.txt"
    },
    "infer_cfg": {
        "n_images_per_class": 1000,
        "format": ["collage", "ind"]
    }
}

_TYPE = {
    "experiment": str,
    "data": {
        "n_classes": int,
    },
    "train": {
        "batch_size": int,
        "unet": {
            "sample_size": int,
            "out_channels": int,
            "layers_per_block": int,
            "block_out_channels": Tuple[int, ...],
            "down_block_types": Tuple[str, ...],
            "up_block_types": Tuple[str, ...],
        }
    },
    "diffusion": {
        "max_time_steps": int,
        "beta_schedule": str,
    },
    "path": {
        "prefix": str,
        "model": str,
        "gen_file": str,
        "ind_path": str,
        "img_lab_path": str,
        "configs": str,
    },
    "infer_cfg": {
        "n_images_per_class": int,
        "format": List[str]
    }
}


def infer(cfg, epoch, store_format=["collage"], net=None):
  device = utils_cc.get_device()

  noise_scheduler = utils_cc.get_noise_scheduler(cfg)
  # noise_scheduler.set_timesteps(
  #     num_inference_steps=cfg["infer_cfg", "num_inference_steps"],
  #     device=device,
  # )

  if net is None:
    net = utils_cc.ClassConditionedUnet(cfg).to(device)
    model_path = utils.get_path(cfg, "model")
    net.load_state_dict(
        torch.load(model_path,
                   map_location=torch.device(utils_cc.get_device())))

  out_channels = cfg["train", "unet", "out_channels"]
  sample_size = cfg["train", "unet", "sample_size"]

  if "ind" in store_format:
    n_images_per_class = cfg["infer_cfg", "n_images_per_class"]
    n_classes = cfg["data", "n_classes"]
    x = torch.randn(n_classes * n_images_per_class, out_channels, sample_size,
                    sample_size).to(device)
    y = torch.tensor([[i] * n_images_per_class for i in range(n_classes)
                     ]).flatten().to(device)

    dataset = data.TensorDataset(x, y)
    batch_size = cfg["train", "batch_size"]
    dataloader = data.DataLoader(dataset, batch_size=batch_size)

    p_bar = utils.get_p_bar(len(dataloader))
    name_img_list = []
    for b_id, (batch_x, batch_y) in enumerate(dataloader):
      for idx, t in enumerate(noise_scheduler.timesteps):
        with torch.no_grad():
          residual = net(batch_x, t, batch_y)
        batch_x = noise_scheduler.step(residual, t, batch_x).prev_sample
        if cfg["args", "debug"] and idx == 20:
          break

      batch_y = batch_y.detach().cpu().numpy()
      batch_x = (batch_x + 1) / 2
      batch_x = batch_x.detach().cpu().numpy()

      for idx, (img, lab) in enumerate(zip(batch_x, batch_y)):
        img_id = b_id * batch_size + idx
        out_file = utils.get_path(cfg,
                                  "ind_path",
                                  class_id=lab,
                                  img_id=img_id,
                                  epoch=epoch + 1)
        if img.shape[0] == 1:
          img = img.squeeze()
        else:
          img = np.transpose(img, (1, 2, 0))
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)
        img.save(out_file)
        name_img_list.append({"filepath": out_file, "class": int(lab)})

      if cfg["args", "debug"]:
        break
      p_bar.update(1)
    img_lab_path = utils.get_path(cfg, "img_lab_path")
    utils.write_json(file_path=img_lab_path, data_dict=name_img_list)

  if "collage" in store_format:
    x = torch.randn(80, out_channels, sample_size, sample_size).to(device)
    y = torch.tensor([[i] * 8 for i in range(10)]).flatten().to(device)

    time_steps = noise_scheduler.timesteps
    p_bar = utils.get_p_bar(len(time_steps))
    for idx, t in enumerate(time_steps):
      with torch.no_grad():
        residual = net(x, t, y)
      x = noise_scheduler.step(residual, t, x).prev_sample
      if cfg["args", "debug"] and idx == 20:
        break
      p_bar.update(1)

    _, ax = plt.subplots(1, 1, figsize=(12, 12))
    x = (x + 1) / 2
    grid = torchvision.utils.make_grid(x.detach().cpu().clip(0, 1), nrow=8)
    grid = grid.permute(1, 2, 0)
    ax.imshow(grid)

    try:
      gen_file = utils.get_path(cfg, "gen_file", epoch=epoch + 1)
    except:
      gen_file = utils.get_path(cfg, "gen_file", epoch=epoch)
    plt.savefig(gen_file)

  p_bar.close()


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

  infer(cfg=cfg, epoch="pred", store_format=cfg["infer_cfg", "format"])


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  main()
