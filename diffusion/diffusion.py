"""Module for diffusion process."""

import sys
from typing import Tuple

import tensorflow as tf

from diffusion import schedule
import utils


class Diffusion:
  """The class encapsulating (reverse) diffusion process.
  
  Attributes:
    max_time_steps: The time step after which the image gets converted to pure
      noise.
    variance_schedule: The type of beta schedule. It's value can be "linear"
      or "cosine".
    pred_type: The value that we are trying to learn either noise or original
      image.
    seed: For random number generation.
    to_enforce_zero_terminal_snr: Whether to enforce zero terminal SNR value
      or not

  Raise:
    ValueError: For invalid variance_schedule values.
  """

  def __init__(self, cfg):
    self.pred_type = cfg["diffusion_cfg", "pred_type"]
    self.max_time_steps = cfg["diffusion_cfg", "max_time_steps"]

    variance_schedule = cfg["diffusion_cfg", "variance_schedule"]
    if variance_schedule == "linear":
      self.beta = schedule.linear(max_steps=self.max_time_steps)
    elif variance_schedule == "cosine":
      self.beta = schedule.cosine(max_steps=self.max_time_steps)
    else:
      raise ValueError(f"{variance_schedule} not supported. Must be `linear`.")
    self.beta = tf.cast(self.beta, utils.get_default_dtype(cfg))

    self.sqrt_beta = tf.sqrt(self.beta)

    self.alpha = 1 - self.beta
    self.inv_sqrt_alpha = 1 / tf.sqrt(self.alpha)

    self.alpha_bar = tf.math.cumprod(self.alpha)
    self.neg_alpha_bar = 1 - self.alpha_bar
    self.inv_sqrt_alpha_bar = 1 / tf.sqrt(self.alpha_bar)

    self.alpha_bar_prev = self.alpha_bar / self.alpha
    self.neg_alpha_bar_prev = 1 - self.alpha_bar_prev
    self.sqrt_alpha_bar_prev = tf.sqrt(self.alpha_bar_prev)

    self.x_0_var = (self.beta * self.neg_alpha_bar_prev) / self.neg_alpha_bar

    self.sqrt_alpha_bar = tf.sqrt(self.alpha_bar)
    self.sqrt_neg_alpha_bar = tf.sqrt(self.neg_alpha_bar)

    self.snr = self.alpha_bar / self.neg_alpha_bar

    self.eps_coeff_ddpm = self.beta / self.sqrt_neg_alpha_bar

    self.x_t_coeff_ddpm = self.neg_alpha_bar_prev * self.sqrt_alpha_bar
    self.x_0_coeff_ddpm = self.beta * self.sqrt_alpha_bar_prev
    self.mean_coeff_ddpm = 1 / self.neg_alpha_bar

    self.rng = tf.random.Generator.from_seed(cfg["seed"])

  def enforce_zero_terminal_snr(self):
    """Enforces zero terminal SNR for the schedule."""
    # TODO: Include this.
    # Convert betas to alphas_bar_sqrt
    alphas = 1 - self.beta
    alphas_bar = tf.math.cumprotf.linspace(
        start=self.beta_init,
        stop=self.beta_final,
        num=self.max_time_steps,
    )
    alphas_bar_sqrt = tf.sqrt(alphas_bar)

    # Store old values.
    alphas_bar_sqrt_0 = tf.identity(alphas_bar_sqrt[0])
    alphas_bar_sqrt_T = tf.identity(alphas_bar_sqrt[-1])

    # Shift so last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T

    # Scale so first timestep is back to old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 -
                                            alphas_bar_sqrt_T)

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt**2
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    alphas = tf.concat([alphas_bar[0:1], alphas], axis=0)
    self.beta = 1 - alphas

  def _gather(self, tf_array: tf.Tensor, indices: tf.Tensor):
    """Returns tensor after `gather`-ing and adding dimensions."""
    return tf.gather(tf_array, indices)[:, tf.newaxis, tf.newaxis, tf.newaxis]

  def get_noise(self, shape: Tuple[int, ...]) -> tf.Tensor:
    """Returns Gaussian noise tensor of given shape."""
    return self.rng.normal(shape=shape, mean=0.0, stddev=1.0)

  def get_mean_and_variance(self, step_t: tf.Tensor) -> tf.Tensor:
    return (self._gather(self.sqrt_alpha_bar,
                         step_t), self._gather(self.sqrt_neg_alpha_bar, step_t))

  def eps_from_x_0(self, x_0: tf.Tensor, x_t: tf.Tensor, step_t: tf.Tensor):
    mean, variance = self.get_mean_and_variance(step_t=step_t)
    return (x_t - mean * x_0) / variance

  def eps_from_vel(self, vel: tf.Tensor, x_t: tf.Tensor, step_t: tf.Tensor):
    mean, variance = self.get_mean_and_variance(step_t=step_t)
    return mean * vel + variance * x_t

  def x_0_from_eps(self, eps: tf.Tensor, x_t: tf.Tensor, step_t: tf.Tensor):
    mean, variance = self.get_mean_and_variance(step_t=step_t)
    return (x_t - variance * eps) / mean

  def x_0_from_vel(self, vel: tf.Tensor, x_t: tf.Tensor, step_t: tf.Tensor):
    mean, variance = self.get_mean_and_variance(step_t=step_t)
    return mean * x_t - variance * vel

  def vel_from_eps_and_x_0(
      self,
      eps: tf.Tensor,
      x_0: tf.Tensor,
      step_t: tf.Tensor,
  ) -> tf.Tensor:
    mean, variance = self.get_mean_and_variance(step_t=step_t)
    return mean * eps - variance * x_0

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
      data: The tuple of x_t and noise or x_0 or velocity.
    """
    # Compute mean component.
    sqrt_alpha_bar_t = self._gather(self.sqrt_alpha_bar, step_t)
    mean_comp = sqrt_alpha_bar_t * x_0
    # Compute variance componenet.
    noise = self.get_noise(shape=x_0.shape)
    sqrt_neg_alpha_bar_t = self._gather(self.sqrt_neg_alpha_bar, step_t)
    var_comp = sqrt_neg_alpha_bar_t * noise
    # Get the final noisy batch.
    x_t = mean_comp + var_comp

    # Form the data.
    if self.pred_type == "eps":
      data = (x_t, noise)
    elif self.pred_type == "x_0":
      data = (x_t, x_0)
    elif self.pred_type == "v":
      vel = self.vel_from_eps_and_x_0(eps=noise, x_0=x_0, step_t=step_t)
      data = (x_t, vel)
    else:
      raise ValueError(f"Invalid pred type: {self.pred_type}")
    return data

  def reverse_step_ddpm(
      self,
      x_t: tf.Tensor,
      model_output: tf.Tensor,
      step_t: tf.Tensor,
  ) -> tf.Tensor:
    """Reverse process of DDPM.
    
    Args:
      x_t: The tensor from which noise is to be removed.
      model_output: The predicted noise.
      step_t: The time step for each image in the batch.
    
    Returns:
      x_t_minus_1: The image after one iteration of noise removal.
    """
    if self.pred_type == "eps":
      pred_eps = model_output
    elif self.pred_type == "x_0":
      pred_eps = self.eps_from_x_0(x_0=model_output, x_t=x_t, step_t=step_t)
    elif self.pred_type == "v":
      pred_eps = self.eps_from_vel(vel=model_output, x_t=x_t, step_t=step_t)
    else:
      raise ValueError(f"Invalid pred type {self.pred_type}.")

    noise = self.get_noise(shape=x_t.shape)
    # Compute the mean component.
    eps_coeff_ddpm_t = self._gather(self.eps_coeff_ddpm, step_t)
    inv_sqrt_alpha_t = self._gather(self.inv_sqrt_alpha, step_t)
    mean_comp = inv_sqrt_alpha_t * (x_t - eps_coeff_ddpm_t * pred_eps)
    # Compute the variance component.
    var_comp = self._gather(self.sqrt_beta, step_t) * noise

    x_t_minus_1 = mean_comp + var_comp
    return x_t_minus_1

  def reverse_step_ddim(
      self,
      x_t: tf.Tensor,
      model_output: tf.Tensor,
      step_t: tf.Tensor,
      step_t_minus_1: tf.Tensor,
  ) -> tf.Tensor:
    """Reverse process of DDIM.
    
    Args:
      x_t: The tensor from which noise is to be removed.
      model_output: The predicted noise or predicted ground-truth.
      step_t: The time step for each image in the batch.
      step_t_minus_1: The previous time_step
    
    Returns:
      x_t_minus_1: The image after one iteration of noise removal.
    """
    sqrt_alpha_bar_t_minus_1 = self._gather(self.sqrt_alpha_bar, step_t_minus_1)
    sqrt_neg_alpha_bar_t_minus_1 = self._gather(self.sqrt_neg_alpha_bar,
                                                step_t_minus_1)

    if self.pred_type == "eps":
      pred_x_0 = self.x_0_from_eps(eps=model_output, x_t=x_t, step_t=step_t)
      pred_eps = model_output
    elif self.pred_type == "x_0":
      pred_x_0 = model_output
      pred_eps = self.eps_from_x_0(x_0=model_output, x_t=x_t, step_t=step_t)
    elif self.pred_type == "v":
      pred_x_0 = self.x_0_from_vel(vel=model_output, x_t=x_t, step_t=step_t)
      pred_eps = self.eps_from_vel(vel=model_output, x_t=x_t, step_t=step_t)
    else:
      raise ValueError(f"Invalid pred type {self.pred_type}.")

    x_t_minus_1 = (sqrt_alpha_bar_t_minus_1 * pred_x_0 +
                   sqrt_neg_alpha_bar_t_minus_1 * pred_eps)
    return x_t_minus_1

  def get_loss_weight(self, step_t):
    snr_t = self._gather(self.snr, step_t)
    return snr_t / (snr_t + 1)

  def get_distillation_target(self, x_t, step_t, x_t_minus_2, step_t_minus_2):
    sqrt_alpha_bar_t = self._gather(self.sqrt_alpha_bar, step_t)
    sqrt_neg_alpha_bar_t = self._gather(self.sqrt_neg_alpha_bar, step_t)

    sqrt_alpha_bar_t_minus_2 = self._gather(self.sqrt_alpha_bar, step_t_minus_2)
    sqrt_neg_alpha_bar_t_minus_2 = self._gather(self.sqrt_neg_alpha_bar,
                                                step_t_minus_2)

    var_ratio = sqrt_neg_alpha_bar_t_minus_2 / sqrt_neg_alpha_bar_t
    num = x_t_minus_2 - var_ratio * x_t
    den = sqrt_alpha_bar_t_minus_2 - var_ratio * sqrt_alpha_bar_t
    pred_x_0 = num / den
    pred_eps = self.eps_from_x_0(x_0=pred_x_0, x_t=x_t, step_t=step_t)

    if self.pred_type == "x_0":
      return pred_x_0
    elif self.pred_type == "eps":
      return pred_eps
    elif self.pred_type == "v":
      return self.vel_from_eps_and_x_0(eps=pred_eps,
                                       x_0=pred_x_0,
                                       step_t=step_t)
    else:
      raise ValueError(f"Invalid pred type {self.pred_type}.")


if __name__ == "__main__":
  sys.exit("Intended for import.")
