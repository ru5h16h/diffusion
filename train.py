#!/usr/bin/env python3
"""Module for training the diffusion model."""

import argparse
import logging

import tensorflow as tf
import tqdm

import configs
from dataset import data_prep
from diffusion import diffusion
import infer
from model import model
import utils

_CFG = {
    "experiment": utils.get_current_ts(),
    "seed": 42,
    "default_dtype": "float32",
    "data_cfg": {
        "dataset": "mnist",
        "img_size": 32,
        "split": "train",
        "data_path": "img_align_celeba.zip",
        "filter_classes": None,
    },
    "diffusion_cfg": {
        'max_time_steps': 1000,
        'sampling_process': 'ddim',
        'inference_steps': 32,
        'pred_type': 'v',
        'variance_schedule': 'cosine',
    },
    "train_cfg": {
        'batch_size': 32,
        'epochs': 100,
        'model': {
            'n_channels': 32,
            'channel_mults': [1, 2, 4, 8],
            'is_attn': [False, False, True, True],
            'out_channels': 1,
            'n_blocks': 2
        },
        'sample_every': 1000,
        'weight_strategy': 'snr',
        "save_at": [50, 100],
        "continue_training": False,
    },
    "infer_cfg": {
        'only_last': True,
        'store_individually': False,
        'store_gif': False,
        'store_collage': True,
        'n_images_approx': 8
    },
    "path": {
        "weights": "runs/{experiment}/weights/model_{epoch}.keras",
        "writer": "runs/{experiment}/writer",
        "gen_dir": "runs/{experiment}/generated_data/{experiment}",
        "pre_trained_weights": "",
        "configs": "runs/{experiment}/configs.json",
    }
}


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--debug",
      action="store_true",
      help="toggles debug mode",
  )
  parser.add_argument(
      "--configs",
      type=str,
      help="path to the configs",
  )
  return parser.parse_args()


def get_weight_t(diff_model: diffusion.Diffusion, step_t: tf.Tensor, cfg):
  """Returns the weight as per given configurations."""
  weight_strategy = cfg["train_cfg", "weight_strategy"]
  if weight_strategy == "const":
    return 1.0
  elif weight_strategy == "snr":
    return diff_model.get_loss_weight(step_t)
  else:
    raise ValueError("Invalid loss strategy.")


def get_start_epoch(cfg):
  if not cfg["train_cfg", "continue_training"]:
    return 0
  pre_trained_path = utils.get_path(cfg, "pre_trained_weights")
  if pre_trained_path:
    return int(pre_trained_path.split("model_")[-1]) - 1
  else:
    return 0


def train(
    tf_dataset: tf.data,
    data_len: int,
    diff_model: diffusion.Diffusion,
    unet_model: model.UNetWithAttention,
    cfg,
) -> None:
  """Trains the diffusion model.

  Args:
    tf_dataset: The dataset used for training.
    diff_model: The object of class diffusion.Diffusion containing various
      function related to diffusion process.
    unet_model: The object of the model to be trained on.
    ckpt_manager: The checkpoint maanger
  """
  # Create summary writer.
  write_dir = utils.get_path(cfg, "writer")
  summary_writer = tf.summary.create_file_writer(write_dir)

  start_epoch = get_start_epoch(cfg)
  epochs = cfg["train_cfg", "epochs"]
  sample_every = cfg["train_cfg", "sample_every"]
  max_t = cfg["diffusion_cfg", "max_time_steps"]

  rng = tf.random.Generator.from_seed(cfg["seed"])
  for epoch in range(start_epoch, epochs):

    p_bar = tqdm.tqdm(total=data_len, position=0, leave=True)
    for idx, batch in enumerate(iter(tf_dataset)):
      # Generate random time steps for each image in the batch.
      step_t = rng.uniform(
          shape=(batch.shape[0],),
          minval=1,
          maxval=max_t,
          dtype=tf.int32,
      )
      # Get noisy data using forward process.
      data = diff_model.forward_process(x_0=batch, step_t=step_t)
      # Perform train step.
      weight_t = get_weight_t(diff_model, step_t, cfg)
      unet_model.train_step(data, step_t, weight_t)

      # Infer after certain steps.
      step = epoch * data_len + idx
      if step % sample_every == 0 and step > 0:
        infer.infer(unet_model, diff_model, cfg, out_file_id=str(step))
      p_bar.update(1)

    loss = unet_model.loss_metric.result()
    with summary_writer.as_default():
      tf.summary.scalar("loss", loss, step=epoch)
    logging.info(f"Average loss for epoch {epoch + 1}/{epochs}: {loss: 0.6f}")

    # Save the model with minimum training loss.
    model_path = utils.get_path(cfg, "weights", epoch=epoch + 1)
    if epoch + 1 in cfg["train_cfg", "save_at"]:
      unet_model.save(model_path)
    unet_model.reset_metric_states()


def main():
  """The entry point of the training."""
  args = parse_args()
  if args.debug:
    _CFG["experiment"] = f"{_CFG['experiment']}_debug"

  cfg = configs.Configs(_CFG, args.configs, args)
  logging.info(f"Experiment: {cfg['experiment']}")

  # Load diffusion model.
  diff_model = diffusion.Diffusion(cfg=cfg)

  # Load UNet model.
  unet_model = model.UNetWithAttention(**cfg["train_cfg", "model"])

  # Load weights, if given
  pre_trained_path = utils.get_path(cfg, "pre_trained_weights")
  if pre_trained_path:
    unet_model.load_weights(pre_trained_path)
    logging.info(f"Continuing from path {pre_trained_path}.")

  # Load dataset.
  tf_dataset, data_len = data_prep.get_datasets(cfg)
  train(tf_dataset, data_len, diff_model, unet_model, cfg)


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  main()
