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
  return tf.as_dtype(configs.cfg["default_dtype"])


if __name__ == "__main__":
  sys.exit("Intended for import.")
