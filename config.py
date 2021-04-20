conf = {
    "WORK_PATH": "/content/zncu/work",
    "CUDA_VISIBLE_DEVICES": "0",# gpu编号  原程序：0,1,2,3
    "data": {
        'dataset_path': "/content/set/Gait",
        'resolution': '64',
        'dataset': 'CASIA-B',
        # In CASIA-B, data of subject #5 is incomplete.
        # Thus, we ignore it in training.
        # For more detail, please refer to
        # function: utils.data_loader.load_data
        'pid_num': 20,
        'pid_shuffle': False,
    },
    "model": {
        'hidden_dim': 256,
        'lr': 1e-4,
        'hard_or_full_trip': 'full',
        'batch_size': (2, 8),
        'restore_iter': 0,
        'total_iter': 30000,
        'margin': 0.2,
        'num_workers': 0,
        'frame_num': 30,
        'model_name': 'GaitSet',
    },
}
