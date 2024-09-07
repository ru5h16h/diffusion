cfg = {
    "data": {
        "set": "cifar10",
    },
    "train": {
        "unet": {
            "sample_size": 32,
            "out_channels": 3,
            "down_block_types": ("DownBlock2D", "DownBlock2D",
                                 "AttnDownBlock2D", "DownBlock2D"),
            "up_block_types":
                ("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
            "block_out_channels": (64, 128, 256, 256),
        },
    }
}
