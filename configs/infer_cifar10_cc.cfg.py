cfg = {
    "train": {
        "unet": {
            "sample_size": 32,
            "out_channels": 3,
            "down_block_types": ("DownBlock2D", "DownBlock2D",
                                 "AttnDownBlock2D", "DownBlock2D"),
            "up_block_types":
                ("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
            "block_out_channels": (32, 64, 128, 128),
        },
    },
    "path": {
        "model":
            "runs_cc/20240907T173222M268/checkpoints/model_500",
        "gen_file":
            "runs_cc/20240907T173222M268/generated_images/{experiment}/plots/{epoch}.png",
        "ind_path":
            "runs_cc/20240907T173222M268/generated_images/{experiment}/ind/images/{img_id}.png",
        "img_lab_path":
            "runs_cc/20240907T173222M268/generated_images/{experiment}/ind/img_lab.json"
    },
}
