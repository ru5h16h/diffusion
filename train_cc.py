import logging

import diffusers
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils import data
import torchvision
import tqdm

import configs
import utils

_CFG = {
    "experiment": utils.get_current_ts(),
    "data": {
        "set": "mnist/",
        "n_classes": 10,
    },
    "train": {
        "batch_size": 128,
        "epochs": 100,
        "lr": 1e-3,
        "save_at": [5, 25, 50, 75, 100],
    },
    "diffusion": {
        "max_time_steps": 1000,
        "beta_schedule": "squaredcos_cap_v2",
        "infer_at": [5, 25, 50, 75, 100],
    },
    "path": {
        "model": "runs/{experiment}/checkpoints/model_{epoch}",
        "gen_file": "runs/{experiment}/generated_images/plot/{epoch}.png",
        "configs": "runs/{experiment}/configs.json",
    }
}


def get_data(cfg):
  dataset = torchvision.datasets.MNIST(
      root=cfg["data", "set"],
      train=True,
      download=True,
      transform=torchvision.transforms.ToTensor(),
  )
  train_dataloader = data.DataLoader(
      dataset,
      batch_size=cfg["train", "batch_size"],
      shuffle=True,
  )
  return train_dataloader


class ClassConditionedUnet(nn.Module):

  def __init__(self, num_classes, class_emb_size=4):
    super().__init__()

    # The embedding layer will map the class label to a vector of size class_emb_size
    self.class_emb = nn.Embedding(num_classes, class_emb_size)

    # Self.model is an unconditional UNet with extra input channels to accept the conditioning information (the class embedding)
    self.model = diffusers.UNet2DModel(
        sample_size=28,
        in_channels=1 + class_emb_size,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(32, 64, 64),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
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
  return diffusers.DDPMScheduler(
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


def infer(noise_scheduler, net, cfg, epoch):
  device = get_device()
  x = torch.randn(80, 1, 28, 28).to(device)
  y = torch.tensor([[i] * 8 for i in range(10)]).flatten().to(device)

  p_bar = utils.get_p_bar(len(noise_scheduler.timesteps))
  for idx, t in enumerate(noise_scheduler.timesteps):
    with torch.no_grad():
      residual = net(x, t, y)
    x = noise_scheduler.step(residual, t, x).prev_sample
    p_bar.update(idx)

  p_bar.close()
  _, ax = plt.subplots(1, 1, figsize=(12, 12))
  ax.imshow(
      torchvision.utils.make_grid(x.detach().cpu().clip(-1, 1), nrow=8)[0])
  # Save the plot as a PNG file
  gen_file = utils.get_path(cfg, "gen_file", epoch=epoch + 1)
  plt.savefig(gen_file)


def main():
  args = utils.parse_args()
  cfg = configs.Configs(_CFG, args.configs, args)
  logging.info(f"Experiment: {cfg['experiment']}")

  device = get_device()

  train_dataloader = get_data(cfg)
  net = ClassConditionedUnet(num_classes=cfg["data", "n_classes"]).to(device)
  loss_fn = nn.MSELoss()
  opt = torch.optim.Adam(net.parameters(), lr=cfg["train", "lr"])
  noise_scheduler = get_noise_scheduler(cfg)

  debug = cfg["args", "debug"]
  max_time_steps = cfg["diffusion", "max_time_steps"]
  n_epochs = cfg["train", "epochs"]
  save_at = cfg["train", "save_at"]
  infer_at = cfg["diffusion", "infer_at"]
  losses = []

  p_bar = utils.get_p_bar(len(train_dataloader))
  for epoch in range(n_epochs):
    for idx, (x, y) in enumerate(train_dataloader):
      x = x.to(device) * 2 - 1
      y = y.to(device)

      noise = torch.randn_like(x)
      timesteps = torch.randint(0, max_time_steps - 1,
                                (x.shape[0],)).long().to(device)
      noisy_x = noise_scheduler.add_noise(x, noise, timesteps)

      pred = net(noisy_x, timesteps, y)
      loss = loss_fn(pred, noise)

      opt.zero_grad()
      loss.backward()
      opt.step()

      losses.append(loss.item())
      p_bar.update(idx)

      if debug and idx > 10:
        break

    avg_loss = sum(losses) / len(losses)
    logging.info(f"Finished epoch {epoch}. Average loss: {avg_loss:05f}")

    if epoch in save_at or debug:
      model_path = utils.get_path(cfg, "model", epoch=epoch)
      torch.save(net.state_dict(), model_path)
    if epoch in infer_at or debug:
      infer(noise_scheduler, net, cfg, epoch)

    if debug and epoch == 0:
      break
  p_bar.close()


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  main()
