# Model configuration


yowo_config = {
    'yowo-d19': {
        # input size
        'train_size': 224,
        'test_size': 224,
        # backbone
        ## 2D
        'backbone_2d': 'yolov2',
        'pretrained_2d': True,
        'stride': 32,
        ## 3D
        'backbone_3d': 'resnext101',
        'pretrained_3d': True,
        # head
        'head_dim': 1024,
        'head_norm': 'BN',
        'head_act': 'lrelu',
        # post process
        'conf_thresh': 0.2,
        'nms_thresh': 0.5,
        # matcher
        'ignore_thresh': 0.5,
        # loss
        'loss_obj_weight': 5.0,
        'loss_noobj_weight': 1.0,
        'loss_cls_weight': 1.0,
        'loss_reg_weight': 1.0,
        # train config
        'batch_size': 8,
        'accumulate': 16,
        'optimizer': 'adamw',
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'max_epoch': 5,
        'lr_epoch': [1, 2, 3, 4],
        'base_lr': 1e-4,
        'lr_decay_ratio': 0.5,
        # warmup strategy
        'warmup': 'linear',
        'warmup_factor': 0.00066667,
        'wp_iter': 500,
    },

}