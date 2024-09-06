import diffusers
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils import data
import torchvision
import tqdm


def get_data():
  dataset = torchvision.datasets.MNIST(
      root="mnist/",
      train=True,
      download=True,
      transform=torchvision.transforms.ToTensor(),
  )
  train_dataloader = data.DataLoader(dataset, batch_size=8, shuffle=True)
  return train_dataloader


class ClassConditionedUnet(nn.Module):

  def __init__(self, num_classes=10, class_emb_size=4):
    super().__init__()

    # The embedding layer will map the class label to a vector of size class_emb_size
    self.class_emb = nn.Embedding(num_classes, class_emb_size)

    # Self.model is an unconditional UNet with extra input channels to accept the conditioning information (the class embedding)
    self.model = diffusers.UNet2DModel(
        sample_size=28,  # the target image resolution
        in_channels=1 +
        class_emb_size,  # Additional input channels for class cond.
        out_channels=1,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(32, 64, 64),
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",  # a regular ResNet upsampling block
        ),
    )

  # Our forward method now takes the class labels as an additional argument
  def forward(self, x, t, class_labels):
    # Shape of x:
    bs, ch, w, h = x.shape

    # class conditioning in right shape to add as additional input channels
    class_cond = self.class_emb(class_labels)  # Map to embedding dimension
    class_cond = class_cond.view(bs, class_cond.shape[1], 1,
                                 1).expand(bs, class_cond.shape[1], w, h)
    # x is shape (bs, 1, 28, 28) and class_cond is now (bs, 4, 28, 28)

    # Net input is now x and class cond concatenated together along dimension 1
    net_input = torch.cat((x, class_cond), 1)  # (bs, 5, 28, 28)

    # Feed this to the UNet alongside the timestep and return the prediction
    return self.model(net_input, t).sample  # (bs, 1, 28, 28)


def get_noise_scheduler():
  return diffusers.DDPMScheduler(
      num_train_timesteps=1000,
      beta_schedule="squaredcos_cap_v2",
  )


def get_device():
  device = "mps"
  if torch.backends.mps.is_available():
    device = "cpu"
  elif torch.cuda.is_available():
    device = "cuda"
  else:
    device = "cpu"
  return device


def infer(noise_scheduler, net):
  device = get_device()
  x = torch.randn(80, 1, 28, 28).to(device)
  y = torch.tensor([[i] * 8 for i in range(10)]).flatten().to(device)

  for i, t in tqdm(enumerate(noise_scheduler.timesteps)):

    # Get model pred
    with torch.no_grad():
      residual = net(x, t, y)  # Again, note that we pass in our labels y

    # Update sample with step
    x = noise_scheduler.step(residual, t, x).prev_sample

  fig, ax = plt.subplots(1, 1, figsize=(12, 12))
  ax.imshow(torchvision.utils.make_grid(x.detach().cpu().clip(-1, 1),
                                        nrow=8)[0],
            cmap="Greys")


def main():
  n_epochs = 10

  device = get_device()

  train_dataloader = get_data()
  net = ClassConditionedUnet().to(device)
  loss_fn = nn.MSELoss()
  opt = torch.optim.Adam(net.parameters(), lr=1e-3)
  noise_scheduler = get_noise_scheduler()

  losses = []
  for epoch in range(n_epochs):
    for x, y in tqdm.tqdm(train_dataloader):
      x = x.to(device) * 2 - 1
      y = y.to(device)

      noise = torch.randn_like(x)
      timesteps = torch.randint(0, 999, (x.shape[0],)).long().to(device)
      noisy_x = noise_scheduler.add_noise(x, noise, timesteps)

      pred = net(noisy_x, timesteps, y)
      loss = loss_fn(pred, noise)

      opt.zero_grad()
      loss.backward()
      opt.step()

      losses.append(loss.item())
    avg_loss = sum(losses[-100:]) / 100
    print(
        f"Finished epoch {epoch}. Average of the last 100 loss values: {avg_loss:05f}"
    )

  infer(noise_scheduler, net)


if __name__ == "__main__":
  main()
