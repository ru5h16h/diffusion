cfg = {
    "data": {
        "set": "cifar10",
        "retrain_classes": [0, 5, 9]
    },
    "train": {
        "epochs":
            500,
        "unet": {
            "sample_size": 32,
            "out_channels": 3,
            "down_block_types": ("DownBlock2D", "DownBlock2D",
                                 "AttnDownBlock2D", "DownBlock2D"),
            "up_block_types":
                ("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
            "block_out_channels": (32, 64, 128, 128),
        },
        "save_at": [
            5, 10, 25, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500
        ],
    },
    "diffusion": {
        "infer_at": [
            1, 5, 10, 25, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500
        ],
    },
    "path": {
        "checkpoint": "runs_cc/20240907T173222M268/checkpoints/model_500"
    }
}
