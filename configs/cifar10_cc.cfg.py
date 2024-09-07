cfg = {
    "data": {
        "set": "cifar10",
    },
    "train": {
        "epochs": 250,
        "unet": {
            "sample_size": 32,
            "out_channels": 3,
            "down_block_types": ("DownBlock2D", "DownBlock2D",
                                 "AttnDownBlock2D", "DownBlock2D"),
            "up_block_types":
                ("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
            "block_out_channels": (32, 64, 128, 128),
        },
        "save_at": [5, 10, 25, 50, 75, 100, 150, 200, 250],
    },
    "diffusion": {
        "infer_at": [1, 5, 10, 25, 50, 75, 100, 150, 200, 250],
    }
}
