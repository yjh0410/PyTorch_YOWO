# Model configuration


yowo_config = {
    'yowo-d19': {
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
    },

}