import tensorflow as tf
import tensorflow_datasets as tfds

import configs
import utils


def configure_for_performance(ds: tf.data):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=1000)
  ds = ds.repeat()
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


def get_datasets():
  # Load the dataset with "train" split only.
  dataset_name = configs.cfg["data_cfg", "dataset"]
  split = configs.cfg["data_cfg", "split"]
  tf_dataset = tfds.load(name=dataset_name, split=split, as_supervised=True)

  # Preprocess data.
  tf_dataset = tf_dataset.map(
      lambda image, _: reshape_and_rescale(image),
      num_parallel_calls=tf.data.AUTOTUNE,
  )
  tf_dataset = configure_for_performance(tf_dataset)
  return tf_dataset
