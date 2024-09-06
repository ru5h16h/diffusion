cfg = {
    "data_cfg": {
        "dataset": "cifar10",
        "filter_classes": [0, 5, 9],
    },
    "train_cfg": {
        "batch_size": 128,
        'epochs': 500,
        "model": {
            'out_channels': 3,
        },
        "save_at": [10, 25, 50, 100, 250, 500],
        'sample_every': 10000,
    },
    "path": {
        "checkpoints_dir": "runs/20240905T234338M475/checkpoints/ckpt-500",
    }
}
