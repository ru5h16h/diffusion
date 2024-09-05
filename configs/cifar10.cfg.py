cfg = {
    "data_cfg": {
        "dataset": "cifar10",
    },
    "train_cfg": {
        'epochs': 500,
        "model": {
            'out_channels': 3,
        },
        "save_at": [50, 100, 250, 500],
        'sample_every': 25000,
    }
}
