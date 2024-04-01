#!/usr/bin/env python3
"""Functions related to evaluating diffusion model"""

import logging

import numpy as np
from scipy import linalg
import tensorflow as tf

import configs
from dataset import data_prep
from diffusion import diffusion
import infer
from model import model


def generate_using_diffusion(
    diff_model: diffusion.Diffusion,
    unet_model: model.UNetWithAttention,
    count: int,
):
  for idx in range(count):
    infer.infer(
        unet_model=unet_model,
        diff_model=diff_model,
        out_file_id=f"eval_{idx}",
    )


class FID:

  def __init__(self):
    self.inception_model = tf.keras.applications.InceptionV3(
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )

  def get_embedding(self, tf_dataset, count):
    image_embeddings = []
    for _ in range(count):
      images = next(iter(tf_dataset))
      embeddings = self.inception_model.predict(images)
      image_embeddings.extend(embeddings)
    return np.array(image_embeddings)

  def calculate_fid(self, gt_embeddings, gen_embeddings):
    # calculate mean and covariance statistics
    mu1, sigma1 = gt_embeddings.mean(axis=0), np.cov(gt_embeddings,
                                                     rowvar=False)
    mu2, sigma2 = gen_embeddings.mean(axis=0), np.cov(gen_embeddings,
                                                      rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
      covmean = covmean.real
      # calculate score
      fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

  def eval(self, diff_model, unet_model):
    batch_size = configs.cfg["train_cfg", "batch_size"]
    count = configs.cfg["eval_cfg", "n_images_approx"] // batch_size
    logging.info(f"Generating {count * batch_size} images.")
    generate_using_diffusion(
        diff_model=diff_model,
        unet_model=unet_model,
        count=count,
    )
    img_size = configs.cfg["eval_cfg", "img_size"]
    gen_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory=configs.cfg["eval_cfg", "gen_dir"],
        labels=None,
        image_size=(img_size, img_size),
        batch_size=batch_size,
    )
    gen_embeddings = self.get_embedding(tf_dataset=gen_dataset, count=count)

    gt_dataset = data_prep.get_datasets(for_eval=True)
    gt_embeddings = self.get_embedding(tf_dataset=gt_dataset, count=count)

    return self.calculate_fid(gt_embeddings, gen_embeddings)


def main():
  configs.cfg = configs.Configs(path="configs.yaml")

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

  fid = FID()
  fid_score = fid.eval(diff_model, unet_model)
  logging.info(f"Final FID score is: {fid_score}.")


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  main()
