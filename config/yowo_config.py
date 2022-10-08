# Model configuration


yowo_config = {
    'yowo': {
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
        # matcher
        'ignore_thresh': 0.5,
        # loss
        'loss_obj_weight': 5.0,
        'loss_noobj_weight': 1.0,
        'loss_cls_weight': 1.0,
        'loss_reg_weight': 1.0,
        # anchor box
        'anchor_size': {
            'ucf24': [[22, 38],
                      [40, 81],
                      [51, 130],
                      [73, 158],
                      [112, 189]], # 224
            'jhmdb21': [[30,  99],
                        [53, 128],
                        [56, 180],
                        [98, 185],
                        [157, 200]], # 224
            'ava_v2.2': [[23, 68],
                         [41, 133],
                         [68, 163],
                         [105, 188],
                         [165, 203]] # 224
        }
    },

    'yowo_nano': {
        # backbone
        ## 2D
        'backbone_2d': 'yolov2',
        'pretrained_2d': True,
        'stride': 32,
        ## 3D
        'backbone_3d': 'shufflenetv2',
        'model_size': '1.0x',
        'pretrained_3d': True,
        # head
        'head_dim': 1024,
        'head_norm': 'BN',
        'head_act': 'lrelu',
        # matcher
        'ignore_thresh': 0.5,
        # loss
        'loss_obj_weight': 5.0,
        'loss_noobj_weight': 1.0,
        'loss_cls_weight': 1.0,
        'loss_reg_weight': 1.0,
        # anchor box
        'anchor_size': {
            'ucf24': [[22, 38],
                      [40, 81],
                      [51, 130],
                      [73, 158],
                      [112, 189]], # 224
            'jhmdb21': [[30,  99],
                        [53, 128],
                        [56, 180],
                        [98, 185],
                        [157, 200]], # 224
            'ava_v2.2': [[23, 68],
                         [41, 133],
                         [68, 163],
                         [105, 188],
                         [165, 203]] # 224
        }
    },
    
}