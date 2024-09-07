cfg = {
    "data": {
        "set": "cifar10",
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
    },
    "train": {
        "unet": {
            "sample_size": 32,
            "out_channels": 3,
        }
    }
}
