#!/usr/bin/env python3
"""Script to generate images."""

import itertools
import logging
import math
import os
import time
from typing import List, Tuple

from PIL import Image
import imageio
import numpy as np
import tensorflow as tf
import tqdm

import configs
from dataset import data_prep
from diffusion import diffusion
from model import model
import utils

SEP_WIDTH = 2


def divide_batch_size(batch_size: int) -> Tuple[int, int]:
  sqrt_batch_size = int(math.sqrt(batch_size))
  for idx in range(sqrt_batch_size, 0, -1):
    if batch_size % idx == 0:
      return idx, batch_size // idx
  raise ValueError(f"Batch size: {batch_size} cannot be divided.")


def get_canvas_dim(sequence, only_last=True):
  batch_size, width, height, _ = sequence[0].shape
  if only_last:
    num_rows, num_cols = divide_batch_size(batch_size)
  else:
    num_rows = batch_size
    num_cols = len(sequence)
  f_width = num_cols * width + (num_cols - 1) * SEP_WIDTH
  f_height = num_rows * height + (num_rows - 1) * SEP_WIDTH
  return f_width, f_height, num_cols, num_rows


def get_coord(idx, jdx, height, width):
  row_s = idx * (height + SEP_WIDTH)
  col_s = jdx * (width + SEP_WIDTH)
  row_e = (idx + 1) * height + idx * SEP_WIDTH
  col_e = (jdx + 1) * width + jdx * SEP_WIDTH
  return row_s, col_s, row_e, col_e


def get_gif_path(step: str) -> str:
  gen_dir = configs.cfg["infer_cfg", "gen_dir"]
  sampling_process = configs.cfg["diffusion_cfg", "sampling_process"]
  gif_dir = os.path.join(gen_dir, "gif", sampling_process)
  os.makedirs(gif_dir, exist_ok=True)
  return os.path.join(gif_dir, f"{step}.gif")


def get_png_path(step: str) -> str:
  gen_dir = configs.cfg["infer_cfg", "gen_dir"]
  sampling_process = configs.cfg["diffusion_cfg", "sampling_process"]
  png_dir = os.path.join(gen_dir, "png", sampling_process)
  os.makedirs(png_dir, exist_ok=True)
  return os.path.join(png_dir, f"{step}.png")


def get_ind_dir() -> str:
  gen_dir = configs.cfg["infer_cfg", "gen_dir"]
  sampling_process = configs.cfg["diffusion_cfg", "sampling_process"]
  ind_dir = os.path.join(gen_dir, "eval", sampling_process)
  os.makedirs(ind_dir, exist_ok=True)
  return ind_dir


def store_gif(sequence: List[np.ndarray], step: str) -> None:
  _, width, height, channels = sequence[0].shape
  f_width, f_height, num_cols, num_rows = get_canvas_dim(sequence=sequence)

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

  imageio.mimsave(gif_path, final_seq, fps=10)


def store_jpeg(sequence: List[np.ndarray], step: str, only_last: bool) -> None:
  _, width, height, _ = sequence[0].shape
  f_sub_img_shape = (height, width, 3)
  f_width, f_height, num_cols, num_rows = get_canvas_dim(
      sequence=sequence,
      only_last=only_last,
  )
  # Create a canvas of required shape.
  canvas = (np.ones((f_height, f_width, 3)) * 255).astype(np.uint8)
  for idx in range(num_rows):
    for jdx in range(num_cols):
      if only_last:
        bdx = idx * num_cols + jdx
        sdx = -1
      else:
        bdx = idx
        sdx = jdx
      image = sequence[sdx][bdx]
      image = np.broadcast_to(image, f_sub_img_shape)
      row_s, col_s, row_e, col_e = get_coord(idx, jdx, height, width)
      canvas[row_s:row_e, col_s:col_e, :] = image

  png_path = get_png_path(step=step)
  imageio.imwrite(png_path, canvas)


def infer(
    unet_model: model.UNetWithAttention,
    diff_model: diffusion.Diffusion,
    inference_steps: int = None,
    out_file_id: str = "predict",
):
  """Inference of diffusion model.
  
  Args:
    diff_model: The object of class diffusion.Diffusion containing various
      function related to diffusion process.
    unet_model: The trained model.
    inference steps: Number of inference steps.
    out_file_id: The ID of output file.
  """
  if not inference_steps:
    # It isn't None when distilling the model.
    inference_steps = configs.cfg["diffusion_cfg", "inference_steps"]
  # TODO: Debug why starting with max_time_steps is not working. It works when
  #   inferences starts with max_time_steps - 1.
  max_t = configs.cfg["diffusion_cfg", "max_time_steps"]
  step_size = max_t // inference_steps
  rem = (max_t - 1) % inference_steps
  init_step_size = step_size + 1 if max_t % inference_steps != 0 else step_size
  step_sizes = [init_step_size] * rem + [step_size] * (inference_steps - rem)
  if max_t % inference_steps == 0:
    if step_sizes[-1] != 1:
      step_sizes[-1] -= 1
    else:
      del step_sizes[-1]
  time_seq = list(itertools.accumulate(step_sizes))

  # Generate noise to infer a given batch.
  shape = utils.get_input_shape()
  model_input = diff_model.get_noise(shape=shape)

  sampling_process = configs.cfg["diffusion_cfg", "sampling_process"]
  bar = tf.keras.utils.Progbar(len(time_seq))
  to_gif = []
  st_time = time.time()
  # Iterate backward in time for reverse process.
  for idx, (ts, ts_size) in enumerate(list(zip(time_seq, step_sizes))[::-1]):
    # Same time steps for all images in batch.
    step_t = tf.fill((shape[0],), ts)
    # Get predicted noise.
    model_output = unet_model(ft=model_input, step_t=step_t)

    # Remove noise.
    if sampling_process == "ddpm":
      model_input = diff_model.reverse_step_ddpm(
          x_t=model_input,
          model_output=model_output,
          step_t=step_t,
      )
    elif sampling_process == "ddim":
      step_t_minus_1 = step_t - ts_size
      model_input = diff_model.reverse_step_ddim(
          x_t=model_input,
          model_output=model_output,
          step_t=step_t,
          step_t_minus_1=step_t_minus_1,
      )
    else:
      raise ValueError(f"Invalid reverse diffusion type: {sampling_process}.")

    # Make list to gif.
    to_gif.append(data_prep.de_normalize(model_input).numpy())
    bar.update(idx + 1)
  time_taken = time.time() - st_time
  logging.info(f"Time taken for generation: {time_taken:0.3f}")

  if configs.cfg["infer_cfg", "store_individually"]:
    ind_dir = get_ind_dir()
    batch = to_gif[-1].astype(np.uint8)
    for idx, image in enumerate(batch):
      image_path = os.path.join(ind_dir, f"{out_file_id}_{idx}.png")
      imageio.imwrite(image_path, image)
  if configs.cfg["infer_cfg", "store_gif"]:
    store_gif(sequence=to_gif, step=out_file_id)
  if configs.cfg["infer_cfg", "store_collage"]:
    store_jpeg(
        sequence=to_gif,
        step=out_file_id,
        only_last=configs.cfg["infer_cfg", "only_last"],
    )
  return time_taken


def main():
  args = utils.parse_args()
  configs.cfg = configs.Configs(path=args.configs)
  logging.info(f"Using configs: {args.configs}.")

  # Load diffusion model.
  seed = configs.cfg["seed"]
  diff_model = diffusion.Diffusion(seed=seed, **configs.cfg["diffusion_cfg"])

  # Load UNet model.
  unet_model = model.UNetWithAttention(**configs.cfg["train_cfg", "model"])

  # Load checkpoint
  ckpt = tf.train.Checkpoint(unet_model=unet_model)
  ckpt_configs = configs.cfg["train_cfg", "checkpoint"]
  ckpt_manager = tf.train.CheckpointManager(checkpoint=ckpt, **ckpt_configs)
  if ckpt_manager.latest_checkpoint:
    # TODO: Resolve the "Value in checkpoint could not be found in the
    #  restored object" warning.
    ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
    logging.info("Restored from {}".format(ckpt_manager.latest_checkpoint))
  else:
    raise ValueError("Checkpoint not present.")

  gen_dir = configs.cfg["infer_cfg", "gen_dir"]
  logging.info(f"Storing generations at {gen_dir}.")

  batch_size = configs.cfg["train_cfg", "batch_size"]
  n_images_approx = configs.cfg["infer_cfg", "n_images_approx"]
  count = math.ceil(n_images_approx / batch_size) or 1
  times = []
  for idx in tqdm.tqdm(range(count)):
    times.append(infer(unet_model, diff_model, out_file_id=f"pred_{idx}"))
  if len(times) > 1:
    times = times[1:]
    avg_time = np.average(times)
    logging.info(f"Average time over {count} inferences: {avg_time:0.3f}.")

  trainable_params_count = unet_model.count_params()
  logging.info(f"Trainable parameter count: {trainable_params_count}.")


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  main()
