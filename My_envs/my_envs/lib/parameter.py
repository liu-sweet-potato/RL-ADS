"""每个参数的取值选项最后一个为-1，表示无效"""
parameters = [
    {
        "z_dim_Omni": [2, 3, 4, -1],
        "window_length_Omni": [50, 100, 150, 200, -1],
        "rnn_num_hidden_Omni": [400, 500, 600, -1],
        "dense_dim_Omni": [400, 500, 600, -1],
        "batch_size_Omni": [50, 75, 100, -1],
        "level_Omni": [0.0025, 0.0050, 0.0075, 0.0100, -1],
    },
    # mtad的参数 —— bs最大256，lookback最大100，避免OOM
    {
        "lookback_mtad": [50, 75, 100, -1],
        "epochs_mtad": [5, 10, 15, -1],
        "gru_n_layers_mtad": [1, 2, 3, -1],
        "gru_hid_dim_mtad": [100, 150, 200, -1],
        "bs_mtad": [64, 128, 256, -1],
        "init_lr_mtad": [1e-5, 1e-4, 1e-3, 1e-2, -1],
    },
    # InterFusion的参数
    {
        "model.window_length_InterFusion": [50, 100, 200, -1],
        "model.z_dim_InterFusion": [2, 3, 4, -1],
        "model.z2_dim_InterFusion": [10, 11, 12, 13, 14, -1],
        "model.l2_reg_InterFusion": [1e-5, 1e-4, 1e-3, -1],
        "train.batch_size_InterFusion": [50, 100, 150, -1],
        "train.max_epoch_InterFusion": [10, 15, 20, 25, -1],
    },
    # SDFVAE
    {
        "learning_rate_SDFVAE": [0.002, 0.0002, 0.00002, -1],
        "s_dims_SDFVAE": [6, 8, 10, -1],
        "d_dims_SDFVAE": [6, 8, 10, 12, -1],
        "conv_dims_SDFVAE": [50, 100, 150, -1],
        "hidden_dims_SDFVAE": [20, 40, 60, -1],
        "epochs_SDFVAE": [40, 50, 60, -1],
    }
]