"""This module contains functions related to variance scheduling."""

import math

import tensorflow as tf

import utils

# Beta min and max values for linear schedule.
LIN_BETA_START = 1e-4
LIN_BETA_END = 0.02
# The constant s in consine schedule.
COS_S = 0.008
# Beta min and max values for cosine schedule
COS_BETA_START = 1e-4
COS_BETA_END = 0.999


def linear(max_steps: int) -> tf.Tensor:
  """Returns linear variance schedule for given maximum time steps."""
  return tf.linspace(start=LIN_BETA_START, stop=LIN_BETA_END, num=max_steps)


def cosine(max_steps):
  """Returns cosine variance schedule for given maximum time steps."""

  def cosine_alpha_bar(step_t):
    """Returns alpha_bar at given time step."""
    return tf.cos((step_t / max_steps + COS_S) / (1 + COS_S) * 0.5 * math.pi)**2

  time_steps = tf.linspace(start=0, stop=max_steps, num=max_steps + 1)
  alpha_bar = cosine_alpha_bar(time_steps) / cosine_alpha_bar(
      tf.zeros_like(time_steps))
  beta = 1 - alpha_bar[1:] / alpha_bar[:-1]
  beta = tf.clip_by_value(beta, COS_BETA_START, COS_BETA_END)
  return beta
