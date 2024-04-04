#!/usr/bin/env python3
"""Progressive distillation of the original model."""

import logging

import tensorflow as tf

import configs
from dataset import data_prep
from diffusion import diffusion
import infer
from model import model
import utils


def train_distill(
    tf_dataset: tf.data,
    diff_model: diffusion.Diffusion,
    teacher_model: model.UNetWithAttention,
):
  """Distills the teacher model.

  Args:
    tf_dataset: The dataset used for training.
    diff_model: The object of class diffusion.Diffusion containing various
      function related to diffusion process.
    teacher_model: The teacher model used for distillation.
  """
  # Load UNet model.
  student_model = model.UNetWithAttention(**configs.cfg["train_cfg", "model"])

  # Create checkpoint manager.
  student_ckpt = tf.train.Checkpoint(unet_model=student_model)
  ckpt_configs = configs.cfg["train_cfg", "checkpoint"]
  student_ckpt_manager = tf.train.CheckpointManager(checkpoint=student_ckpt,
                                                    **ckpt_configs)
  logging.info(f"Checkpoint dir: {ckpt_configs['directory']}")

  shape = utils.get_input_shape()
  # TODO: Debug: ValueError: Weights for model 'time_nn_init' have not yet
  #   been created. Weights are created when the model is first called on
  #   inputs or `build()` is called with an `input_shape`, even when calling
  #   get_weights on teacher model.
  teacher_model(tf.zeros(shape), tf.zeros((shape[0])))
  student_model(tf.zeros(shape), tf.zeros((shape[0])))
  student_model.set_weights(teacher_model.get_weights())

  # Initialize teacher and student step count.
  max_t = configs.cfg["diffusion_cfg", "max_time_steps"]
  teacher_steps = configs.cfg["diffusion_cfg", "inference_steps"]
  teacher_step_size = max_t // teacher_steps
  student_steps = teacher_steps // 2

  # Create summary writer.
  logs_dir = configs.cfg["train_cfg", "train_logs_dir"]
  summary_writer = tf.summary.create_file_writer(logs_dir)

  epochs = configs.cfg["train_cfg", "epochs"]
  sample_every = configs.cfg["train_cfg", "sample_every"]
  patience = configs.cfg["train_cfg", "patience"]

  data_len = len(tf_dataset)
  while student_steps >= 4:

    min_loss = float("inf")
    for epoch in range(epochs):

      bar = tf.keras.utils.Progbar(data_len - 1)
      for idx, batch in enumerate(iter(tf_dataset)):
        # Generate random time steps for each image in the batch.
        step_t = tf.random.uniform(
            shape=(batch.shape[0],),
            minval=2 * teacher_step_size,
            maxval=max_t,
            dtype=tf.int32,
        )
        step_t_minus_1 = step_t - 1 * teacher_step_size
        step_t_minus_2 = step_t - 2 * teacher_step_size

        # Get noised data using forward process.
        x_t, _ = diff_model.forward_process(x_0=batch, step_t=step_t)
        # Perform 2 steps of DDIM
        teacher_out_t = teacher_model(ft=x_t, step_t=step_t)
        x_t_minus_1 = diff_model.reverse_step_ddim(
            x_t=x_t,
            model_output=teacher_out_t,
            step_t=step_t,
            step_t_minus_1=step_t_minus_1,
        )
        teacher_out_t_minus_1 = teacher_model(
            ft=x_t_minus_1,
            step_t=step_t_minus_1,
        )
        x_t_minus_2 = diff_model.reverse_step_ddim(
            x_t=x_t_minus_1,
            model_output=teacher_out_t_minus_1,
            step_t=step_t_minus_1,
            step_t_minus_1=step_t_minus_2,
        )
        # Get target for distillation.
        distillation_target = diff_model.get_distillation_target(
            x_t=x_t,
            step_t=step_t,
            x_t_minus_2=x_t_minus_2,
            step_t_minus_2=step_t_minus_2,
        )
        step_t_minus_2 = step_t_minus_2[:, tf.newaxis, tf.newaxis, tf.newaxis]
        target = tf.where(step_t_minus_2 == 0, x_t_minus_2, distillation_target)
        data = (x_t, target)
        # Perform train step for student model.
        student_model.train_step(data, step_t)

        # Infer after appropriate steps.
        step = epoch * data_len + idx
        if step % sample_every == 0 and step > 0:
          logging.info(f"Step: {step}. student_steps: {student_steps}.")
          infer.infer(
              unet_model=student_model,
              diff_model=diff_model,
              out_file_id=f"{student_steps}_{step}",
              inference_steps=student_steps,
          )

        bar.update(idx)

      loss = student_model.loss_metric.result()
      with summary_writer.as_default():
        tf.summary.scalar("loss", loss, step=epoch)
      logging.info(f"Average loss for epoch {epoch + 1}/{epochs}: {loss: 0.4f}")

      # Save the model with minimum training loss.
      # TODO: Do this based on validation score.
      if loss < min_loss:
        student_ckpt_manager.save(checkpoint_number=epoch)
        min_loss = loss
        stop_ctr = 0
      else:
        stop_ctr += 1
      if stop_ctr == patience:
        logging.info("Reached training saturation.")
        break
      student_model.reset_metric_states()

    teacher_steps //= 2
    teacher_step_size = max_t // teacher_steps
    student_steps = teacher_steps // 2
    student_ckpt.restore(student_ckpt_manager.latest_checkpoint)
    teacher_model.set_weights(student_model.get_weights())


def main():
  """Entry point of Progressive dsitillation."""
  configs.cfg = configs.Configs(path="configs.yaml")
  configs.cfg.dump_config()

  # Load diffusion model.
  seed = configs.cfg["seed"]
  diff_model = diffusion.Diffusion(seed=seed, **configs.cfg["diffusion_cfg"])

  # Load UNet model.
  teacher_model = model.UNetWithAttention(**configs.cfg["train_cfg", "model"])

  # Load teacher model checkpoint
  teacher_ckpt = tf.train.Checkpoint(unet_model=teacher_model)
  ckpt_configs = configs.cfg["train_cfg", "teacher_checkpoint"]
  teacher_ckpt_manager = tf.train.CheckpointManager(checkpoint=teacher_ckpt,
                                                    **ckpt_configs)
  if teacher_ckpt_manager.latest_checkpoint:
    # TODO: Resolve the "Value in checkpoint could not be found in the
    #  restored object" warning.
    teacher_ckpt.restore(
        teacher_ckpt_manager.latest_checkpoint).expect_partial()
    logging.info("Restored from {}".format(
        teacher_ckpt_manager.latest_checkpoint))
  else:
    raise ValueError("Checkpoint not present.")

  tf_dataset = data_prep.get_datasets()
  train_distill(tf_dataset, diff_model, teacher_model)


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  main()
