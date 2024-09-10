import matplotlib.pyplot as plt

import configs
import utils

_CFG = {
    "experiment": "mnist_100",
    "title": "MNIST (after 100 epochs)",
    "class_percent": [(0, 9.87), (1, 10.12), (2, 9.92), (3, 9.99), (4, 9.81),
                      (5, 9.84), (6, 9.97), (7, 10.23), (8, 10.03), (9, 10.22)],
    "path": {
        "out_path": "plots/class_balance/{experiment}.png"
    },
    "retrian_classes": [0, 5, 9]
}


def main():
  args = utils.parse_args()
  cfg = configs.Configs(_CFG, args.configs, args, False)
  classes, percent = list(zip(*cfg["class_percent"]))

  plt.style.use("seaborn-v0_8-darkgrid")
  plt.bar(classes, percent)
  plt.xlabel("Classes")
  plt.ylabel("Percent")
  plt.title(cfg["title"])
  plt.xticks(ticks=range(len(classes)), labels=classes)
  plt.savefig(utils.get_path(cfg, "out_path"))


if __name__ == "__main__":
  main()
