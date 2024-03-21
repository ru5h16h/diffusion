import einops
import math
import tensorflow as tf
from tensorflow.keras import layers

from model import activations

DEFAULT_DTYPE = tf.float32


class Identity(layers.Layer):

  def __init__(self):
    super(Identity, self).__init__()

  def call(self, ft, time_emb=None):
    return tf.identity(ft)


class Upsample(layers.Layer):

  def __init__(self, n_channels):
    super(Upsample, self).__init__()
    self.up_sample = layers.Conv2DTranspose(
        filters=n_channels,
        kernel_size=4,
        strides=2,
        padding='SAME',
    )

  def call(self, ft, time_channels=None):
    return self.up_sample(ft)


class Downsample(layers.Layer):

  def __init__(self, n_channels):
    super(Downsample, self).__init__()
    self.down_sample = layers.Conv2D(filters=n_channels,
                                     kernel_size=3,
                                     strides=2,
                                     padding='same')

  def call(self, ft, time_channels=None):
    return self.down_sample(ft)


class Conv2DBlock(layers.Layer):

  def __init__(self, filters, dropout=None, groups=8):
    super(Conv2DBlock, self).__init__()
    self.conv2d = layers.Conv2D(filters, kernel_size=3, padding="same")
    self.norm = layers.GroupNormalization(groups, epsilon=1e-05)
    self.act_fn = activations.SiLU()
    if dropout:
      self.dropout = layers.Dropout(rate=dropout)
    else:
      self.dropout = None

  def call(self, ft, scale_shift=None, training=True):
    ft = self.norm(ft, training=training)
    if scale_shift:
      scale, shift = scale_shift
      ft = ft * (scale + 1) + shift
    ft = self.act_fn(ft)
    if self.dropout is not None:
      ft = self.dropout(ft)
    ft = self.conv2d(ft)
    return ft


class ResidualBlockWithTimeEmbedding(layers.Layer):

  def __init__(self, in_channels, out_channels, groups=8, dropout=0.1):
    super(ResidualBlockWithTimeEmbedding, self).__init__()
    self.time_nn_final = tf.keras.Sequential(
        layers=[
            activations.SiLU(),
            layers.Dense(units=in_channels * 2),
        ],
        name="time_nn_init",
    )

    self.block_1 = Conv2DBlock(filters=out_channels, groups=groups)
    self.block_2 = Conv2DBlock(filters=out_channels,
                               groups=groups,
                               dropout=dropout)

    if in_channels != out_channels:
      self.res_conv = layers.Conv2D(
          filters=out_channels,
          kernel_size=1,
          strides=1,
      )
    else:
      self.res_conv = Identity()

  def call(self, ft, time_emb, training=True):
    time_emb = self.time_nn_final(time_emb, training=training)
    time_emb = einops.rearrange(time_emb, 'b c -> b 1 1 c')
    scale_shift = tf.split(time_emb, num_or_size_splits=2, axis=-1)

    hid = self.block_1(ft=ft, scale_shift=scale_shift, training=training)
    hid = self.block_2(ft=hid, training=training)
    return hid + self.res_conv(ft)


class TimeEmbedding(layers.Layer):

  def __init__(self, n_channels, max_positions=10000):
    super(TimeEmbedding, self).__init__()
    self.n_channels = n_channels
    self.max_position = max_positions

    self.dense_1 = layers.Dense(units=self.n_channels)
    self.act_fn = activations.GELU()
    self.dense_2 = layers.Dense(units=self.n_channels)

  def call(self, pos):
    half_dim = self.n_channels // 8

    emb = math.log(self.max_position) / (half_dim - 1)
    emb = tf.exp(tf.range(half_dim, dtype=tf.float32) * -emb)
    emb = tf.cast(pos, dtype=DEFAULT_DTYPE)[:, None] * emb[None, :]

    emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)

    emb = self.dense_1(emb)
    emb = self.act_fn(emb)
    emb = self.dense_2(emb)
    return emb


class PreGroupNorm(layers.Layer):

  def __init__(self, func, groups=1):
    super(PreGroupNorm, self).__init__()
    self.func = func
    self.norm = layers.GroupNormalization(groups, epsilon=1e-05)

  def call(self, x):
    x = self.norm(x)
    x = self.func(x)
    return x


class Residual(layers.Layer):

  def __init__(self, func):
    super(Residual, self).__init__()
    self.func = func

  def call(self, x, *args, **kwargs):
    return x + self.func(x, *args, **kwargs)


class Attention(layers.Layer):

  def __init__(self, n_channels, n_heads=1, n_dim_heads=None, n_group=32):
    super(Attention, self).__init__()
    if n_dim_heads is None:
      n_dim_heads = n_channels

    self.norm = layers.GroupNormalization(n_group, epsilon=1e-05)
    self.to_qkv = layers.Dense(units=n_heads * n_dim_heads * 3)
    self.to_out = layers.Dense(units=n_channels)

    self.scale = n_dim_heads**-0.5
    self.n_heads = n_heads
    self.n_dim_heads = n_dim_heads

  def call(self, ft, training=True):
    batch, height, width, n_channels = ft.shape
    ft = einops.rearrange(ft, "b h w c -> b (h w) c")
    qkv = self.to_qkv(ft, training=training)

    qkv = einops.rearrange(qkv, "b x (h d) -> b x h d", h=self.n_heads)
    query, key, value = tf.split(qkv, num_or_size_splits=3, axis=-1)

    attn = tf.einsum('b x h d, b y h d -> b x y h', query, key) * self.scale
    attn = tf.nn.softmax(attn, axis=2)

    res = tf.einsum('b x y h, b y h d -> b x h d', attn, value)

    res = einops.rearrange(res, "b x h d -> b x (h d)", h=self.n_heads)
    res = self.to_out(res, training=training)

    res += ft
    res = einops.rearrange(res, "b (h w) c -> b h w c", h=height)
    return res


class UpDownBlock(layers.Layer):

  def __init__(self, in_channels, out_channels, time_channels, has_attn):
    super(UpDownBlock, self).__init__()

    self.res_block = ResidualBlockWithTimeEmbedding(in_channels, out_channels)
    if has_attn:
      self.attn = Residual(PreGroupNorm(Attention(out_channels)))
    else:
      self.attn = Identity()

  def call(self, ft, time_emb):
    ft = self.res_block(ft, time_emb)
    ft = self.attn(ft)
    return ft


class MiddleBlock(layers.Layer):

  def __init__(self, n_channels):
    super(MiddleBlock, self).__init__()

    self.res_block_1 = ResidualBlockWithTimeEmbedding(n_channels, n_channels)
    self.attn = Attention(n_channels)
    self.res_block_2 = ResidualBlockWithTimeEmbedding(n_channels, n_channels)

  def call(self, ft, time_emb):
    ft = self.res_block_1(ft, time_emb)
    ft = self.attn(ft)
    ft = self.res_block_2(ft, time_emb)
    return ft
