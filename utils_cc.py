import diffusers
import torch
from torch import nn


class ClassConditionedUnet(nn.Module):

  def __init__(self, cfg, class_emb_size=4):
    super().__init__()

    # The embedding layer will map the class label to a vector of size class_emb_size
    self.class_emb = nn.Embedding(cfg["data", "n_classes"], class_emb_size)

    # Self.model is an unconditional UNet with extra input channels to accept the conditioning information (the class embedding)
    out_channels = cfg["train", "unet", "out_channels"]
    self.model = diffusers.UNet2DModel(
        in_channels=out_channels + class_emb_size,
        **cfg["train", "unet"],
    )

  def forward(self, x, t, class_labels):
    bs, ch, w, h = x.shape

    # class conditioning in right shape to add as additional input channels
    class_cond = self.class_emb(class_labels)
    class_cond = class_cond.view(bs, class_cond.shape[1], 1,
                                 1).expand(bs, class_cond.shape[1], w, h)
    # x is shape (bs, 1, 28, 28) and class_cond is now (bs, 4, 28, 28)

    # Net input is now x and class cond concatenated together along dimension 1
    net_input = torch.cat((x, class_cond), 1)  # (bs, 5, 28, 28)

    # Feed this to the UNet alongside the timestep and return the prediction
    return self.model(net_input, t).sample  # (bs, 1, 28, 28)


def get_noise_scheduler(cfg):
  return diffusers.DDIMScheduler(
      num_train_timesteps=cfg["diffusion", "max_time_steps"],
      beta_schedule=cfg["diffusion", "beta_schedule"],
  )


def get_device():
  if torch.backends.mps.is_available():
    device = "mps"
  elif torch.cuda.is_available():
    device = "cuda"
  else:
    device = "cpu"
  return device
