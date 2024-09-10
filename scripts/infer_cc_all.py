import copy
import logging
import os

import configs
import infer_cc
import utils

_CFG = {
    "base_exp": "20240906T234605M545",
    "fine_tuned_exp": "20240907T091926M647",
    "dataset": "mnist",
    # "base_exp": "20240907T173222M268",
    # "fine_tuned_exp": "20240908T101533M577",
    # "dataset": "cifar10",
}

MNIST_CFG = {
    "experiment": "{infer_id}",
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
        "model":
            "runs_cc/{exp_id}/checkpoints/model_{epoch}",
        "gen_file":
            "runs_cc/{exp_id}/generated_images/{infer_id}/plots/pred.png",
        "ind_path":
            "runs_cc/{exp_id}/generated_images/{infer_id}/ind/images/{{img_id}}.png",
        "img_lab_path":
            "runs_cc/{exp_id}/generated_images/{infer_id}/ind/img_lab.json"
    },
    "infer_cfg": {
        "n_images_per_class": 1000,
        "format": ["collage", "ind"]
    }
}

CIFAR10_CFG = {
    "experiment": "{infer_id}",
    "data": {
        "n_classes": 10,
    },
    "train": {
        "batch_size": 128,
        "unet": {
            "layers_per_block": 2,
            "sample_size": 32,
            "out_channels": 3,
            "down_block_types": ("DownBlock2D", "DownBlock2D",
                                 "AttnDownBlock2D", "DownBlock2D"),
            "up_block_types":
                ("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
            "block_out_channels": (32, 64, 128, 128),
        }
    },
    "diffusion": {
        "max_time_steps": 1000,
        "beta_schedule": "squaredcos_cap_v2",
    },
    "path": {
        "model":
            "runs_cc/{exp_id}/checkpoints/model_{epoch}",
        "gen_file":
            "runs_cc/{exp_id}/generated_images/{infer_id}/plots/pred.png",
        "ind_path":
            "runs_cc/{exp_id}/generated_images/{infer_id}/ind/images/{{img_id}}.png",
        "img_lab_path":
            "runs_cc/{exp_id}/generated_images/{infer_id}/ind/img_lab.json"
    },
    "infer_cfg": {
        "n_images_per_class": 1000,
        "format": ["collage", "ind"]
    }
}


def update_configs(cfg, exp_id, epoch):
  infer_id = utils.get_current_ts()
  cfg["path", "model"] = cfg["path", "model"].format(exp_id=exp_id, epoch=epoch)
  cfg["experiment"] = cfg["experiment"].format(infer_id=infer_id)
  kwargs = {"infer_id": infer_id, "exp_id": exp_id}
  for path in ["gen_file", "ind_path", "img_lab_path"]:
    cfg["path", path] = cfg["path", path].format(**kwargs)
  return infer_id


def main():
  args = utils.parse_args()
  cfg = configs.Configs(_CFG, save_configs=False)

  if cfg["dataset"] == "mnist":
    data_cfg = MNIST_CFG
  else:
    data_cfg = CIFAR10_CFG

  exp_ids = []
  infer_ids = []
  epoch_cfg = []
  exp_id = cfg["base_exp"]
  model_dir = os.path.join("runs_cc", exp_id,
                           "checkpoints").format(exp_id=exp_id)
  max_epoch = -1
  for model_name in os.listdir(model_dir):
    if not model_name.startswith("model"):
      logging.info(f"Found unexpected model file: {model_name}.")
      continue
    epoch = int(model_name.split("_")[-1])
    if epoch > max_epoch:
      max_epoch = epoch
  if max_epoch == -1:
    raise ValueError("Base model not found.")
  base_cfg = configs.Configs(
      copy.deepcopy(data_cfg),
      args=args,
      save_configs=False,
  )
  infer_id = update_configs(base_cfg, exp_id, max_epoch)
  logging.info("-" * 79)
  logging.info(
      f"Infering epoch 0. exp ID: {exp_id}. infer ID: {infer_id}. cfg: {base_cfg}"
  )
  infer_cc.infer(base_cfg, "pred", base_cfg["infer_cfg", "format"])
  exp_ids.append(exp_id)
  epoch_cfg.append([0])
  infer_ids.append([infer_id])

  epoch_cfg.append([])
  infer_ids.append([])
  exp_id = cfg["fine_tuned_exp"]
  model_dir = os.path.join("runs_cc", exp_id,
                           "checkpoints").format(exp_id=exp_id)
  for model_name in sorted(os.listdir(model_dir),
                           key=lambda x: int(x.split("_")[-1])):
    epoch = int(model_name.split("_")[-1])
    ft_cfg = configs.Configs(
        copy.deepcopy(data_cfg),
        args=args,
        save_configs=False,
    )
    infer_id = update_configs(ft_cfg, exp_id, epoch)
    logging.info("-" * 79)
    logging.info(
        f"Infering epoch {epoch}. exp ID: {exp_id}. infer ID: {infer_id}. cfg: {ft_cfg}"
    )
    infer_cc.infer(ft_cfg, "pred", ft_cfg["infer_cfg", "format"])
    infer_ids[1].append(infer_id)
    epoch_cfg[1].append(epoch)
  exp_ids.append(exp_id)

  logging.info(f"infer_ids: {infer_ids}")
  logging.info(f"epoch_cfg: {epoch_cfg}")
  logging.info(f"exp_ids: {exp_ids}")


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  main()
