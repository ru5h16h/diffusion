import os
import shutil
import zipfile

import tensorflow as tf
import tensorflow_datasets as tfds

import configs
import utils


def configure_for_performance(ds: tf.data):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=1000)
  ds = ds.batch(configs.cfg["train_cfg", "batch_size"])
  ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
  return ds


def normalize(image: tf.Tensor) -> tf.Tensor:
  # Apply standard normalization.
  dtype = utils.get_default_dtype()
  return tf.cast(image, dtype) / 127.5 - 1


def reshape_and_rescale(image: tf.Tensor) -> tf.Tensor:
  # Reshape image if required.
  img_size = configs.cfg["data_cfg", "img_size"]
  height, width, _ = image.shape
  if (height, width) != (img_size, img_size):
    image = tf.image.resize(image, (img_size, img_size))
  image = normalize(image)
  return image


def de_normalize(image: tf.Tensor) -> tf.Tensor:
  return tf.cast((image + 1) * 127.5, tf.uint8)


def get_celeb_a(zip_file_path: str) -> tf.data:
  zip_dir = os.path.dirname(zip_file_path)
  with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
    data_dir = os.path.join(zip_dir, os.path.splitext(zip_ref.filename)[0])
    if not os.path.exists(data_dir):
      zip_ref.extractall(zip_dir)

  file_names = os.listdir(data_dir)
  eval_samples = 10000
  train_samples = len(file_names) - eval_samples

  train_dir = os.path.join(data_dir, "train")
  if (not os.path.exists(train_dir) or
      len(os.listdir(train_dir)) != train_samples):
    os.makedirs(train_dir, exist_ok=True)
    for idx in range(1, train_samples + 1):
      file_name = f"{idx:06d}.jpg"
      src = os.path.join(data_dir, file_name)
      dst = os.path.join(train_dir, file_name)
      shutil.move(src, dst)

  eval_dir = os.path.join(data_dir, "eval")
  if not os.path.exists(eval_dir) or len(os.listdir(eval_dir)) != eval_samples:
    os.makedirs(eval_dir, exist_ok=True)
    for idx in range(train_samples + 1, train_samples + eval_samples + 1):
      file_name = f"{idx:06d}.jpg"
      src = os.path.join(data_dir, file_name)
      dst = os.path.join(eval_dir, file_name)
      shutil.move(src, dst)

  img_size = configs.cfg["data_cfg", "img_size"]
  tf_dataset = tf.keras.preprocessing.image_dataset_from_directory(
      directory=train_dir,
      labels=None,
      image_size=(img_size, img_size),
      batch_size=None,
  )
  tf_dataset = tf_dataset.map(
      lambda image: normalize(image),
      num_parallel_calls=tf.data.AUTOTUNE,
  )
  return tf_dataset


def get_datasets():
  # Load the dataset with "train" split only.
  dataset_name = configs.cfg["data_cfg", "dataset"]
  split = configs.cfg["data_cfg", "split"]
  if dataset_name == "celeb_a":
    data_path = configs.cfg["data_cfg", "data_path"]
    tf_dataset = get_celeb_a(zip_file_path=data_path)
    data_len = len(tf_dataset)
  else:
    tf_dataset = tfds.load(name=dataset_name, split=split, as_supervised=True)
    filter_classes = configs.cfg["data_cfg", "filter_classes"]
    tf_dataset = tf_dataset.filter(lambda image, label: tf.reduce_any(
        [tf.math.equal(label, cl) for cl in filter_classes]))
    data_len = len([_ for _ in tf_dataset])
    # Preprocess data.
    tf_dataset = tf_dataset.map(
        lambda image, _: reshape_and_rescale(image),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
  tf_dataset = configure_for_performance(tf_dataset)
  return tf_dataset, data_len
