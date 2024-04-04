#!/usr/bin/env python3
"""Module for training the diffusion model."""

import logging

import tensorflow as tf

import configs
from dataset import data_prep
from diffusion import diffusion
import infer
from model import model


def get_weight_t(diff_model: diffusion.Diffusion, step_t: tf.Tensor):
  """Returns the weight as per given configurations."""
  weight_strategy = configs.cfg["train_cfg", "weight_strategy"]
  if weight_strategy == "const":
    return 1.0
  elif weight_strategy == "snr":
    return diff_model.get_loss_weight(step_t)
  else:
    raise ValueError("Invalid loss strategy.")


def train(
    tf_dataset: tf.data,
    diff_model: diffusion.Diffusion,
    unet_model: model.UNetWithAttention,
    ckpt_manager: tf.train.CheckpointManager,
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
  logs_dir = configs.cfg["train_cfg", "train_logs_dir"]
  summary_writer = tf.summary.create_file_writer(logs_dir)

  epochs = configs.cfg["train_cfg", "epochs"]
  sample_every = configs.cfg["train_cfg", "sample_every"]
  patience = configs.cfg["train_cfg", "patience"]
  max_t = configs.cfg["diffusion_cfg", "max_time_steps"]

  rng = tf.random.Generator.from_seed(configs.cfg["seed"])
  data_len = len(tf_dataset)
  min_loss = float("inf")
  for epoch in range(epochs):

    bar = tf.keras.utils.Progbar(len(tf_dataset))
    for idx, batch in enumerate(iter(tf_dataset)):
      # Generate random time steps for each image in the batch.
      step_t = rng.uniform(
          shape=(batch.shape[0],),
          minval=1,
          maxval=max_t + 1,
          dtype=tf.int32,
      )
      # Get noisy data using forward process.
      data = diff_model.forward_process(x_0=batch, step_t=step_t)
      # Perform train step.
      weight_t = get_weight_t(diff_model, step_t)
      unet_model.train_step(data, step_t, weight_t)

      # Infer after certain steps.
      step = epoch * data_len + idx
      if step % sample_every == 0 and step > 0:
        infer.infer(unet_model, diff_model, out_file_id=str(step))
      bar.update(idx)

    loss = unet_model.loss_metric.result()
    with summary_writer.as_default():
      tf.summary.scalar("loss", loss, step=epoch)
    logging.info(f"Average loss for epoch {epoch + 1}/{epochs}: {loss: 0.4f}")

    # Save the model with minimum training loss.
    # TODO: Do this based on validation score.
    if loss < min_loss:
      ckpt_manager.save(checkpoint_number=epoch)
      min_loss = loss
      stop_ctr = 0
    else:
      stop_ctr += 1
    if stop_ctr == patience:
      logging.info("Reached training saturation.")
      break
    unet_model.reset_metric_states()


def main():
  """The entry point of the training."""
  configs.cfg = configs.Configs(path="configs.yaml")
  configs.cfg.dump_config()

  # Load diffusion model.
  seed = configs.cfg["seed"]
  diff_model = diffusion.Diffusion(seed=seed, **configs.cfg["diffusion_cfg"])

  # Load UNet model.
  unet_model = model.UNetWithAttention(**configs.cfg["train_cfg", "model"])

  # Create checkpoint manager.
  ckpt = tf.train.Checkpoint(unet_model=unet_model)
  ckpt_configs = configs.cfg["train_cfg", "checkpoint"]
  ckpt_manager = tf.train.CheckpointManager(checkpoint=ckpt, **ckpt_configs)
  logging.info(f"Checkpoint dir: {ckpt_configs['directory']}")

  # Load dataset.
  tf_dataset = data_prep.get_datasets()
  train(tf_dataset, diff_model, unet_model, ckpt_manager)


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  main()
