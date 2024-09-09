cfg = {
    "path": {
        "gen_dir":
            "runs/20240908T152926M616/generated_data/{experiment}",
        "checkpoints_dir":
            "runs/20240908T152926M616/checkpoints/ckpt-5",
        "configs":
            "runs/20240908T152926M616/generated_data/{experiment}/configs.json",
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
