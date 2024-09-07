cfg = {
    "data_cfg": {
        "dataset": "cifar10",
    },
    "train_cfg": {
        'epochs': 250,
        "model": {
            'out_channels': 3,
        },
        "save_at": [5, 10, 25, 50, 75, 100, 150, 200, 250],
        "infer_at": [1, 5, 10, 25, 50, 75, 100, 150, 200, 250],
    }
}
