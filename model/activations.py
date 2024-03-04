import tensorflow as tf

from tensorflow.keras import layers


class SiLU(layers.Layer):

  def __init__(self):
    super(SiLU, self).__init__()

  def call(self, x):
    return x * tf.nn.sigmoid(x)


class GELU(layers.Layer):

  def __init__(self, approximate=False):
    super(GELU, self).__init__()
    self.approximate = approximate

  def call(self, x):
    if self.approximate:
      coeff = tf.cast(0.044715, x.dtype)
      return 0.5 * x * (1.0 + tf.tanh(0.7978845608028654 *
                                      (x + coeff * tf.pow(x, 3))))
    else:
      return 0.5 * x * (1.0 +
                        tf.math.erf(x / tf.cast(1.4142135623730951, x.dtype)))
