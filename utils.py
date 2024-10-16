"""Common utility functions."""

import argparse
import datetime
import functools
import json
import os
import sys
import time

# import tensorflow as tf
import tqdm


def profile(func):
  """A decorator for profiling time of a function."""

  def wrapper(*args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    print(f"{func.__name__} took {end_time - start_time} seconds to execute.")
    return result

  return wrapper


def get_default_dtype(cfg):
  """Returns the default data type specified in the configurations."""
  return tf.as_dtype(cfg["default_dtype"])


def get_input_shape(cfg, batch_size=None):
  """Returns the shape of image batch that will be using in training."""
  if not batch_size:
    batch_size = cfg["train_cfg", "batch_size"]
  img_size = cfg["data_cfg", "img_size"]
  img_channels = cfg["train_cfg", "model", "out_channels"]
  return (batch_size, img_size, img_size, img_channels)


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
  return parser.parse_known_args()


def get_current_ts() -> str:
  now = datetime.datetime.now()
  timestamp = now.strftime("%Y%m%dT%H%M%S")
  ms = now.microsecond // 1000
  return f"{timestamp}M{ms:03d}"


@functools.lru_cache
def cached_makedirs(dir_path):
  if not dir_path:
    return
  os.makedirs(dir_path, exist_ok=True)


def get_path(cfg, key, **kwargs):
  if "experiment" not in kwargs:
    kwargs["experiment"] = cfg["experiment"]
  path = cfg["path", key].format(**kwargs)
  cached_makedirs(os.path.dirname(path))
  return path


def safe_open(path, mode):
  if mode not in {"wb", "w"}:
    raise ValueError("Only use safe open in write mode.")
  os.makedirs(os.path.dirname(path), exist_ok=True)
  return open(path, mode)


def write_json(file_path: str, data_dict):
  with safe_open(file_path, "w") as fp:
    json.dump(data_dict, fp)


def get_p_bar(length):
  return tqdm.tqdm(total=length, position=0, leave=True)


def write_txt(file_path, data):
  with safe_open(file_path, "w") as fp:
    fp.write(str(data))


if __name__ == "__main__":
  sys.exit("Intended for import.")
