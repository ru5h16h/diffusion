#!/usr/bin/env python3
"""Module for training the diffusion model."""

import logging

import tensorflow as tf

import configs
from dataset import data_prep
from diffusion import diffusion
import infer
from model import model
import utils


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
    data_len: int,
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

  if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.latest_checkpoint.split(sep='ckpt-')[-1])
  else:
    start_epoch = 0
  epochs = configs.cfg["train_cfg", "epochs"]
  sample_every = configs.cfg["train_cfg", "sample_every"]
  patience = configs.cfg["train_cfg", "patience"]
  precision = configs.cfg["train_cfg", "precision"]
  max_t = configs.cfg["diffusion_cfg", "max_time_steps"]

  rng = tf.random.Generator.from_seed(configs.cfg["seed"])
  min_loss = float("inf")
  prev_loss = float("inf")
  for epoch in range(start_epoch, epochs):

    bar = tf.keras.utils.Progbar(data_len)
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
    logging.info(f"Average loss for epoch {epoch + 1}/{epochs}: {loss: 0.6f}")

    # Save the model with minimum training loss.
    # TODO: Do this based on validation score.
    if loss < min_loss and prev_loss - loss > precision:
      ckpt_manager.save(checkpoint_number=epoch)
      min_loss = loss
      stop_ctr = 0
    else:
      stop_ctr += 1
    if patience != -1 and stop_ctr == patience:
      logging.info("Reached training saturation.")
      break
    prev_loss = loss
    unet_model.reset_metric_states()


def main():
  """The entry point of the training."""
  args = utils.parse_args()
  configs.cfg = configs.Configs(path=args.configs)
  logging.info(f"Using configs: {args.configs}.")
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
  if ckpt_manager.latest_checkpoint:
    # TODO: Resolve the "Value in checkpoint could not be found in the
    #  restored object" warning.
    ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
    logging.info("Restored from {}".format(ckpt_manager.latest_checkpoint))
  else:
    logging.info("Starting training from scratch.")
  logging.info(f"Checkpoint dir: {ckpt_configs['directory']}")

  # Load dataset.
  tf_dataset, data_len = data_prep.get_datasets()
  train(tf_dataset, data_len, diff_model, unet_model, ckpt_manager)


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  main()
