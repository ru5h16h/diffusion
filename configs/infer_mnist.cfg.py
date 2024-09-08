cfg = {
    "path": {
        "gen_dir":
            "runs/20240907T232257M584/generated_data/{experiment}",
        "checkpoints_dir":
            "runs/20240907T232257M584/checkpoints/ckpt-100",
        "configs":
            "runs/20240907T232257M584/generated_data/{experiment}/configs.json",
    },
    "train_cfg": {
        "model": {
            "out_channels": 1,
        }
    },
    "infer_cfg": {
        "store_individually": True,
        "store_collage": False,
        "n_images": 10000,
    }
}
