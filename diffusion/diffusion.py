"""Module for diffusion process."""

import sys
from typing import Tuple

import tensorflow as tf


class Diffusion:
  """The class encapsulating (reverse) diffusion process.
  
  Attributes:
    beta_init: The starting value of beta parameter.
    beta_final: The final value of beta parameter.
    max_time_steps: The time step after which the image gets converted to pure
      noise.
    beta_schedule: The type of beta schedule. It's value can be "linear".

  Raise:
    ValueError: For invalid beta_schedule values.
  """

  def __init__(
      self,
      beta_init: float = 1e-4,
      beta_final: float = 0.02,
      max_time_steps: int = 1000,
      beta_schedule: str = "linear",
  ):
    self.beta_init = tf.constant(beta_init)
    self.beta_final = tf.constant(beta_final)
    self.max_time_steps = max_time_steps

    if beta_schedule == "linear":
      self.beta = tf.linspace(
          start=self.beta_init,
          stop=self.beta_final,
          num=self.max_time_steps,
      )
    else:
      raise ValueError(f"{beta_schedule} not supported. Must be `linear`.")
    self.sqrt_beta = tf.sqrt(self.beta)

    self.alpha = 1 - self.beta
    self.inv_sqrt_alpha = 1 / tf.sqrt(self.alpha)

    self.alpha_bar = tf.math.cumprod(self.alpha)
    self.alpha_bar = tf.tensor_scatter_nd_update(
        self.alpha_bar,
        indices=[[0]],
        updates=[1],
    )

    self.sqrt_alpha_bar = tf.sqrt(self.alpha_bar)
    self.neg_sqrt_alpha_bar = tf.sqrt(1 - self.alpha_bar)

    self.pred_coeff_ddpm = self.beta / self.neg_sqrt_alpha_bar

  def _gather(self, tf_array: tf.Tensor, indices: tf.Tensor):
    """Returns tensor after `gather`-ing and adding dimensions."""
    tf_array_at_t = tf.gather(tf_array, indices)
    return tf_array_at_t[:, tf.newaxis, tf.newaxis, tf.newaxis]

  def get_noise(self, shape: Tuple[int, ...]) -> tf.Tensor:
    """Returns Gaussian noise tensor of given shape."""
    return tf.random.normal(shape=shape, mean=0, stddev=1)

  def forward_process(
      self,
      x_0: tf.Tensor,
      step_t: tf.Tensor,
  ) -> Tuple[tf.Tensor, tf.Tensor]:
    """The forward process of diffusion model to generate noisy image.
      
    Args:
      x_0: The input image batch to which the noise is to be added.
      step_t: The time step for each image in the batch.
    
    Returns:
      x_t: The generate noisy image batch.
      noise: The added noise.
    """
    # Compute mean component.
    sqrt_alpha_bar = self._gather(self.sqrt_alpha_bar, step_t)
    mean_comp = sqrt_alpha_bar * x_0

    # Compute variance componenet.
    noise = self.get_noise(shape=x_0.shape)
    neg_sqrt_alpha_bar = self._gather(self.neg_sqrt_alpha_bar, step_t)
    var_comp = neg_sqrt_alpha_bar * noise

    # Get the final noisy batch.
    x_t = mean_comp + var_comp
    return x_t, noise

  def reverse_process_ddpm(
      self,
      x_t: tf.Tensor,
      pred_noise: tf.Tensor,
      step_t: tf.Tensor,
  ) -> tf.Tensor:
    """Reverse process of DDPM.
    
    Args:
      x_t: The tensor from which noise is to be removed.
      pred_noise: The predicted noise.
      step_t: The time step for each image in the batch.
    
    Returns:
      x_t_minus_1: The image after one iteration of noise removal.
    """
    # Compute the mean component.
    inv_sqrt_alpha = self._gather(self.inv_sqrt_alpha, step_t)
    coeff = self._gather(self.pred_coeff_ddpm, step_t)
    mean_comp = inv_sqrt_alpha * (x_t - coeff * pred_noise)

    # Compute the variance component.
    sqrt_beta = self._gather(self.sqrt_beta, step_t)
    noise = self.get_noise(shape=x_t.shape)
    var_comp = sqrt_beta * noise

    x_t_minus_1 = mean_comp + var_comp
    return x_t_minus_1

  def reverse_process_ddim(
      self,
      x_t: tf.Tensor,
      pred_noise: tf.Tensor,
      step_t: tf.Tensor,
      step_t_minus_1: tf.Tensor,
  ) -> tf.Tensor:
    """Reverse process of DDIM.
    
    Args:
      x_t: The tensor from which noise is to be removed.
      pred_noise: The predicted noise.
      step_t: The time step for each image in the batch.
      step_t_minus_1: The previous time_step
    
    Returns:
      x_t_minus_1: The image after one iteration of noise removal.
    """
    # TODO: Merge this with DDPM.
    sqrt_alpha_bar = self._gather(self.sqrt_alpha_bar, step_t)
    neg_sqrt_alpha_bar = self._gather(self.neg_sqrt_alpha_bar, step_t)

    sqrt_alpha_bar_t_minus_1 = self._gather(self.sqrt_alpha_bar, step_t_minus_1)
    neg_sqrt_alpha_bar_t_minus_1 = self._gather(self.neg_sqrt_alpha_bar,
                                                step_t_minus_1)

    x_t_minus_1 = x_t - neg_sqrt_alpha_bar * pred_noise
    x_t_minus_1 *= sqrt_alpha_bar_t_minus_1 / sqrt_alpha_bar
    x_t_minus_1 += neg_sqrt_alpha_bar_t_minus_1 * pred_noise
    return x_t_minus_1


if __name__ == "__main__":
  sys.exit("Intended for import.")
