import tensorflow as tf
from tensorflow.keras import layers as tf_layers

from model import activations
from model import layers


class UNetWithAttention(tf.keras.Model):

  def __init__(
      self,
      n_channels=64,
      channel_mults=[1, 2, 4],
      is_attn=[False, True, True],
      input_channels=1,
      n_blocks=2,
  ):
    super(UNetWithAttention, self).__init__()
    self.n_blocks = n_blocks
    self.is_attn = is_attn
    self.n_channels = n_channels
    self.n_res = len(channel_mults)
    self.channel_mults = channel_mults
    # Number of channels present in middle block.
    self.mid_channels = self.channel_mults[-1] * self.n_channels

    # Define NN to get the time embeddings
    self.time_channels = n_channels * 4
    self.time_nn = layers.TimeEmbedding(n_channels=self.time_channels)

    # Pass through initial conv layer to update the number of channels for
    # further simplicity.
    self.init_conv = tf_layers.Conv2D(
        filters=n_channels,
        kernel_size=3,
        padding='same',
    )

    # Get all the layers of UNet.
    self.encoder_layers = self.get_encoder_layers()
    self.middle_layer = layers.MiddleBlock(n_channels=self.mid_channels)
    self.decoder_layers = self.get_decoder_layers()

    # Final layers of UNet.
    self.norm = tf_layers.GroupNormalization(groups=8, epsilon=1e-05)
    self.act_fn = activations.SiLU()
    self.final_conv = tf_layers.Conv2D(
        filters=input_channels,
        kernel_size=3,
        padding="same",
    )

    # Define optimizer for training process.
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=tf.Variable(1e-4))

  def get_encoder_layers(self):
    # Create empty list of list to store encoder layers.
    encoder_layers = [[] for _ in range(self.n_res)]
    in_channels = out_channels = self.n_channels
    # Iterate through all the resolutions.
    for idx in range(self.n_res):
      # Skip downsampling for first index.
      # TODO: Add assertion for first resolution to be 1.
      if idx == 0:
        down_sample = layers.Identity()
      else:
        down_sample = layers.Downsample(out_channels)
      encoder_layers[idx].append(down_sample)

      # Increase the number of channels and add blocks of convs.
      out_channels = self.n_channels * self.channel_mults[idx]
      for _ in range(self.n_blocks):
        down_block = layers.UpDownBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            time_channels=self.time_channels,
            has_attn=self.is_attn[idx],
        )
        encoder_layers[idx].append(down_block)
        in_channels = out_channels
    return encoder_layers

  def get_decoder_layers(self):
    decoder_layers = [[] for _ in range(self.n_res)]
    in_channels = self.mid_channels
    for idx, rev_idx in enumerate(reversed(range(self.n_res - 1))):

      out_channels = self.n_channels * self.channel_mults[rev_idx]
      decoder_layers[idx].append(layers.Upsample(out_channels))

      for _ in range(self.n_blocks):
        up_block = layers.UpDownBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            time_channels=self.time_channels,
            has_attn=self.is_attn[rev_idx],
        )
        decoder_layers[idx].append(up_block)
        in_channels = out_channels

    return decoder_layers

  def call(self, ft, time_steps=[1], training=True, **kwargs):
    # input_layer = tf.keras.Input(shape=(32, 32, 1))
    # ft = self.init_conv(input_layer)

    time_emb = self.time_nn(time_steps)

    ft = self.init_conv(ft)

    to_store = []
    for enc_layers in self.encoder_layers:
      for layer in enc_layers:
        if isinstance(layer, layers.Downsample):
          to_store.append(ft)
        ft = layer(ft, time_emb)

    ft = self.middle_layer(ft, time_emb)

    for dec_layers in self.decoder_layers:
      for layer in dec_layers:
        if isinstance(layer, layers.Upsample):
          ft = layer(ft, time_emb)
          ft = tf.concat([ft, to_store.pop()], axis=-1)
        else:
          ft = layer(ft, time_emb)

    ft = self.norm(ft)
    ft = self.act_fn(ft)
    ft = self.final_conv(ft)

    # model = tf.keras.Model(inputs=input_layer, outputs=ft)
    # ft = model(x)
    # model.summary()
    return ft

  def loss_fn(self, real_noise, pred_noise):
    return tf.math.reduce_mean((real_noise - pred_noise)**2)

  @tf.function
  def train_step(self, data, time_steps):
    image_batch, noise_batch = data

    with tf.GradientTape() as tape:
      pred_noise = self(ft=image_batch, time_steps=time_steps, training=True)
      loss = self.loss_fn(real_noise=noise_batch, pred_noise=pred_noise)

    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    return loss
