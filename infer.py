"""Script to generate images."""

import logging
import math
import os
from typing import List, Tuple, Union

from PIL import Image
import imageio
import numpy as np
import tensorflow as tf

import configs
from dataset import data_prep
from diffusion import diffusion
from model import model

SEP_WIDTH = 2


def divide_batch_size(batch_size: int) -> Tuple[int, int]:
  sqrt_batch_size = int(math.sqrt(batch_size))
  for idx in range(sqrt_batch_size, 0, -1):
    if batch_size % idx == 0:
      return idx, batch_size // idx
  raise ValueError(f"Batch size: {batch_size} cannot be divided.")


def get_canvas_dim(img_batch):
  batch_size, width, height, _ = img_batch.shape
  num_rows, num_cols = divide_batch_size(batch_size)
  f_width = num_cols * width + (num_cols - 1) * SEP_WIDTH
  f_height = num_rows * height + (num_rows - 1) * SEP_WIDTH
  return f_width, f_height, num_cols, num_rows


def get_coord(idx, jdx, height, width):
  row_s = idx * (height + SEP_WIDTH)
  col_s = jdx * (width + SEP_WIDTH)
  row_e = (idx + 1) * height + idx * SEP_WIDTH
  col_e = (jdx + 1) * width + jdx * SEP_WIDTH
  return row_s, col_s, row_e, col_e


def get_gif_path(step: Union[str, int]) -> str:
  gif_dir = configs.cfg["infer_cfg", "gif_dir"]
  os.makedirs(gif_dir, exist_ok=True)
  return os.path.join(gif_dir, f"{step}.gif")


def get_png_path(step: Union[str, int]) -> str:
  png_dir = configs.cfg["infer_cfg", "png_dir"]
  os.makedirs(png_dir, exist_ok=True)
  return os.path.join(png_dir, f"{step}.png")


def store_gif(sequence: List[np.ndarray], step: Union[str, int]) -> None:
  _, width, height, channels = sequence[0].shape
  f_width, f_height, num_cols, num_rows = get_canvas_dim(img_batch=sequence[0])

  # Process each image batch in the sequence.
  final_seq = []
  for image_batch in sequence:
    # Create a canvas of required shape.
    canvas = Image.new("RGB", (f_width, f_height), color="white")
    # Add all the images in a batch to the canvas.
    for idx in range(num_rows):
      for jdx in range(num_cols):
        bdx = idx * num_cols + jdx
        image = image_batch[bdx]
        if channels == 1:
          image = image[:, :, 0]
        image = Image.fromarray(image).convert("RGB")
        row_s, col_s, _, _ = get_coord(idx, jdx, height, width)
        canvas.paste(image, (col_s, row_s))
    final_seq.append(canvas)

  gif_path = get_gif_path(step=step)
  imageio.mimsave(gif_path, final_seq, fps=10)


def store_jpeg(img_batch: np.ndarray, step: Union[str, int]) -> None:
  _, width, height, _ = img_batch.shape
  f_sub_img_shape = (height, width, 3)
  f_width, f_height, num_cols, num_rows = get_canvas_dim(img_batch=img_batch)
  # Create a canvas of required shape.
  canvas = (np.ones((f_height, f_width, 3)) * 255).astype(np.uint8)
  for idx in range(num_rows):
    for jdx in range(num_cols):
      bdx = idx * num_cols + jdx
      image = img_batch[bdx]
      image = np.broadcast_to(image, f_sub_img_shape)
      row_s, col_s, row_e, col_e = get_coord(idx, jdx, height, width)
      canvas[row_s:row_e, col_s:col_e, :] = image

  png_path = get_png_path(step=step)
  imageio.imwrite(png_path, canvas)


def infer(
    unet_model: model.UNetWithAttention,
    diff_model: diffusion.Diffusion,
    img_size: int = None,
    batch_size: int = None,
    step: Union[str, int] = "predict",
):
  # Get the shape of the prediction.
  if not img_size:
    img_size = configs.cfg["data_cfg", "img_size"]
  if not batch_size:
    batch_size = configs.cfg["train_cfg", "batch_size"]
  img_channels = configs.cfg["data_cfg", "img_channels"]
  shape = (batch_size, img_size, img_size, img_channels)

  # Generate noise to infer a given batch.
  model_input = diff_model.get_noise(shape=shape)

  max_time_steps = configs.cfg["diffusion_cfg", "max_time_steps"]
  reverse_type = configs.cfg["diffusion_cfg", "reverse_type"]
  inference_steps = configs.cfg["diffusion_cfg", "inference_steps"]

  step_size = max_time_steps // inference_steps
  time_seq = list(range(1, max_time_steps, step_size))

  bar = tf.keras.utils.Progbar(len(time_seq))
  to_gif = []
  # Iterate backward in time for reverse process.
  for idx, rev_ts in enumerate(reversed(time_seq)):
    # Same time steps for all images in batch.
    step_t = tf.fill((batch_size,), rev_ts)
    # Get predicted noise.
    pred_noise = unet_model(model_input, time_steps=step_t)
    # Remove noise.
    if reverse_type == "ddpm":
      model_input = diff_model.reverse_process_ddpm(
          x_t=model_input,
          pred_noise=pred_noise,
          step_t=step_t,
      )
    elif reverse_type == "ddim":
      if rev_ts == 1:
        bar.update(idx + 1)
        continue
      step_t_minus_1 = step_t - step_size
      model_input = diff_model.reverse_process_ddim(
          x_t=model_input,
          pred_noise=pred_noise,
          step_t=step_t,
          step_t_minus_1=step_t_minus_1,
      )
    # Make list to gif.
    to_gif.append(data_prep.de_normalize(model_input).numpy())
    bar.update(idx + 1)

  store_gif(sequence=to_gif, step=step)
  store_jpeg(img_batch=to_gif[-1], step=step)


def main():
  configs.cfg = configs.Configs(path="configs.yaml")

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
  ckpt_manager = tf.train.CheckpointManager(
      checkpoint=ckpt,
      directory=checkpoint_dir,
      max_to_keep=configs.cfg["train_cfg", "checkpoint", "max_to_keep"],
  )
  if ckpt_manager.latest_checkpoint:
    # TODO: Resolve the "Value in checkpoint could not be found in the
    #  restored object" warning.
    ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
    logging.info("Restored from {}".format(ckpt_manager.latest_checkpoint))
  else:
    raise ValueError("Checkpoint not present.")

  infer(unet_model=unet_model, diff_model=diff_model)


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  main()
