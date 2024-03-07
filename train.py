import logging
import os

import tensorflow as tf

import configs
from dataset import data_prep
from diffusion import diffusion
from model import model

import infer


def train(
    tf_dataset: tf.data,
    diff_model: diffusion.Diffusion,
    unet_model: model.UNetWithAttention,
    ckpt_manager,
):
  data_len = len(tf_dataset)
  epochs = configs.cfg["train_cfg", "epochs"]
  sample_every = configs.cfg["train_cfg", "sample_every"]
  max_time_steps = configs.cfg["diffusion_cfg", "max_time_steps"]

  logs_dir = configs.cfg["train_cfg", "train_logs_dir"]
  summary_writer = tf.summary.create_file_writer(logs_dir)

  for epoch in range(epochs):
    bar = tf.keras.utils.Progbar(len(tf_dataset) - 1)
    for idx, batch in enumerate(iter(tf_dataset)):
      # Generate random time steps for each image in the batch.
      time_steps = tf.random.uniform(
          shape=(batch.shape[0],),
          minval=1,
          maxval=max_time_steps,
          dtype=tf.int32,
      )
      # Get noised data using forward process.
      data = diff_model.forward_process(x_0=batch, step_t=time_steps)
      # Perform train step.
      unet_model.train_step(data=data, time_steps=time_steps)

      # Infer after appropriate steps.
      step = epoch * data_len + idx
      if step % sample_every == 0 and step > 0:
        infer.infer(unet_model=unet_model, diff_model=diff_model, step=step)

      bar.update(idx)

    loss = unet_model.loss_metric.result()
    with summary_writer.as_default():
      tf.summary.scalar("loss", loss, step=epoch)
    logging.info(f"Average loss for epoch {epoch + 1}/{epochs}: {loss: 0.4f}")

    ckpt_manager.save(checkpoint_number=epoch)
    unet_model.reset_metric_states()


def main():
  configs.cfg = configs.Configs(path="configs.yaml")
  configs.cfg.dump_config()

  # Load diffusion model.
  max_time_steps = configs.cfg["diffusion_cfg", "max_time_steps"]
  diff_model = diffusion.Diffusion(max_time_steps=max_time_steps)

  # Load UNet model.
  unet_model = model.UNetWithAttention(
      input_channels=configs.cfg["data_cfg", "img_channels"],
      channel_mults=configs.cfg["train_cfg", "model", "channel_mults"],
      is_attn=configs.cfg["train_cfg", "model", "is_attn"],
      n_blocks=configs.cfg["train_cfg", "model", "n_blocks"],
  )

  ckpt = tf.train.Checkpoint(unet_model=unet_model)
  checkpoint_dir = configs.cfg["train_cfg", "checkpoint", "dir"]
  logging.info(f"Storing checkpoint in {checkpoint_dir}")
  os.makedirs(configs.cfg["train_cfg", "checkpoint", "dir"], exist_ok=True)
  ckpt_manager = tf.train.CheckpointManager(
      checkpoint=ckpt,
      directory=checkpoint_dir,
      max_to_keep=configs.cfg["train_cfg", "checkpoint", "max_to_keep"],
  )
  if ckpt_manager.latest_checkpoint:
    # TODO: Implement this.
    pass
  else:
    logging.info("Initializing from scratch.")

  # Load dataset.
  tf_dataset = data_prep.get_datasets()
  train(
      tf_dataset=tf_dataset,
      unet_model=unet_model,
      diff_model=diff_model,
      ckpt_manager=ckpt_manager,
  )


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  main()
