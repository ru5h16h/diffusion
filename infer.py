#!/usr/bin/env python3
"""Script to generate images."""

import itertools
import logging
import math
import os
import time
from typing import List, Tuple

import imageio
import numpy as np
from PIL import Image
import tensorflow as tf
import tqdm

import configs
from dataset import data_prep
from diffusion import diffusion
from model import model
import utils

SEP_WIDTH = 2

_CFG = {
    "experiment": utils.get_current_ts(),
    "seed": 42,
    "default_dtype": "float32",
    "path": {
        "gen_dir":
            "runs/20240905T234338M475/generated_data/{experiment}",
        "checkpoints_dir":
            "runs/20240905T234338M475/checkpoints/ckpt-500",
        "configs":
            "runs/20240905T234338M475/generated_data/{experiment}/configs.json",
    },
    "diffusion_cfg": {
        'max_time_steps': 1000,
        'sampling_process': 'ddim',
        'inference_steps': 32,
        'pred_type': 'v',
        'variance_schedule': 'cosine',
    },
    "data_cfg": {
        "img_size": 32,
    },
    "train_cfg": {
        'model': {
            'n_channels': 32,
            'channel_mults': [1, 2, 4, 8],
            'is_attn': [False, False, True, True],
            'out_channels': 3,
            'n_blocks': 2
        },
        'batch_size': 32
    },
    'only_last': True,
    'store_individually': False,
    'store_gif': False,
    'store_collage': True,
    'n_images_approx': 8,
}


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


def get_gif_path(step: str, cfg) -> str:
  gen_dir = utils.get_path(cfg, "gen_dir")
  sampling_process = cfg["diffusion_cfg", "sampling_process"]
  gif_dir = os.path.join(gen_dir, "gif", sampling_process)
  os.makedirs(gif_dir, exist_ok=True)
  return os.path.join(gif_dir, f"{step}.gif")


def get_png_path(step: str, cfg) -> str:
  gen_dir = utils.get_path(cfg, "gen_dir")
  sampling_process = cfg["diffusion_cfg", "sampling_process"]
  png_dir = os.path.join(gen_dir, "png", sampling_process)
  os.makedirs(png_dir, exist_ok=True)
  return os.path.join(png_dir, f"{step}.png")


def get_ind_dir(cfg) -> str:
  gen_dir = utils.get_path(cfg, "gen_dir")
  sampling_process = cfg["diffusion_cfg", "sampling_process"]
  ind_dir = os.path.join(gen_dir, "eval", sampling_process)
  os.makedirs(ind_dir, exist_ok=True)
  return ind_dir


def store_gif(sequence: List[np.ndarray], step: str, cfg) -> None:
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

  gif_path = get_gif_path(step, cfg)
  imageio.mimsave(gif_path, final_seq, fps=10)


def store_jpeg(sequence: List[np.ndarray], step: str, only_last: bool,
               cfg) -> None:
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

  png_path = get_png_path(step, cfg)
  imageio.imwrite(png_path, canvas)


def infer(
    unet_model: model.UNetWithAttention,
    diff_model: diffusion.Diffusion,
    cfg: configs.Configs,
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
    inference_steps = cfg["diffusion_cfg", "inference_steps"]
  # TODO: Debug why starting with max_time_steps is not working. It works when
  #   inferences starts with max_time_steps - 1.
  max_t = cfg["diffusion_cfg", "max_time_steps"]
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
  shape = utils.get_input_shape(cfg)
  model_input = diff_model.get_noise(shape=shape)

  sampling_process = cfg["diffusion_cfg", "sampling_process"]
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

  if cfg["store_individually"]:
    ind_dir = get_ind_dir(cfg)
    batch = to_gif[-1].astype(np.uint8)
    for idx, image in enumerate(batch):
      image_path = os.path.join(ind_dir, f"{out_file_id}_{idx}.png")
      imageio.imwrite(image_path, image)
  if cfg["store_gif"]:
    store_gif(sequence=to_gif, step=out_file_id, cfg=cfg)
  if cfg["store_collage"]:
    store_jpeg(
        sequence=to_gif,
        step=out_file_id,
        only_last=cfg["only_last"],
        cfg=cfg,
    )
  return time_taken


def main():
  args = utils.parse_args()
  cfg = configs.Configs(_CFG, args.configs, args)
  logging.info(f"Experiment: {cfg['experiment']}")
  logging.info(f"Using configs: {args.configs}.")

  # Load diffusion model.
  diff_model = diffusion.Diffusion(cfg=cfg)

  # Load UNet model.
  unet_model = model.UNetWithAttention(**cfg["train_cfg", "model"])

  # Restore checkpoints.
  ckpt = tf.train.Checkpoint(unet_model=unet_model)
  pre_trained_path = utils.get_path(cfg, "checkpoints_dir")
  if pre_trained_path:
    ckpt.restore(pre_trained_path).expect_partial()
    logging.info(f"Continuing from path {pre_trained_path}.")
  else:
    raise ValueError("Checkpoint path needed.")

  gen_dir = utils.get_path(cfg, "gen_dir")
  logging.info(f"Storing generations at {gen_dir}.")

  batch_size = cfg["train_cfg", "batch_size"]
  n_images_approx = cfg["n_images_approx"]
  count = math.ceil(n_images_approx / batch_size) or 1
  times = []
  for idx in tqdm.tqdm(range(count)):
    times.append(infer(unet_model, diff_model, cfg, out_file_id=f"pred_{idx}"))
  if len(times) > 1:
    times = times[1:]
    avg_time = np.average(times)
    logging.info(f"Average time over {count} inferences: {avg_time:0.3f}.")

  trainable_params_count = unet_model.count_params()
  logging.info(f"Trainable parameter count: {trainable_params_count}.")


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  main()
