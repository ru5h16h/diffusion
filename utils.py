"""Common utility functions."""

import sys
import time

import tensorflow as tf

import configs


def profile(func):
  """A decorator for profiling time of a function."""

  def wrapper(*args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    print(f"{func.__name__} took {end_time - start_time} seconds to execute.")
    return result

  return wrapper


def get_default_dtype():
  """Returns the default data type specified in the configurations."""
  return tf.as_dtype(configs.cfg["default_dtype"])


def get_input_shape():
  """Returns the shape of image batch that will be using in training."""
  batch = configs.cfg["train_cfg", "batch_size"]
  img_size = configs.cfg["data_cfg", "img_size"]
  img_channels = configs.cfg["train_cfg", "model", "out_channels"]
  return (batch, img_size, img_size, img_channels)


if __name__ == "__main__":
  sys.exit("Intended for import.")
