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
        # post process
        'conf_thresh': 0.01,
        'nms_thresh': 0.5,
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
            'ava': []
        }
    },

    'yowo_v2': {
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
        'conf_thresh': 0.01,
        'nms_thresh': 0.5,
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
            'ava': []
        }
    },

    'yowo_v3': {
        # backbone
        ## 2D
        'backbone_2d': 'yolov3',
        'pretrained_2d': True,
        'stride': [8, 16, 32],
        ## 3D
        'backbone_3d': 'resnext101',
        'pretrained_3d': True,
        # head
        'head_dim': [256, 512, 1024],
        'head_norm': 'BN',
        'head_act': 'lrelu',
        # post process
        'conf_thresh': 0.01,
        'nms_thresh': 0.5,
        # matcher
        'ignore_thresh': 0.5,
        # loss
        'loss_obj_weight': 5.0,
        'loss_noobj_weight': 1.0,
        'loss_cls_weight': 1.0,
        'loss_reg_weight': 1.0,
        # anchor box
        'anchor_size': {
            'ucf24': [[19, 30],
                      [32, 58],
                      [39, 94],
                      [61, 92],
                      [48, 133],
                      [62, 172],
                      [78, 137],
                      [89, 188],
                      [129, 199]], # 224
            'jhmdb21': [[26, 73],
                        [35, 114],
                        [42, 158],
                        [64, 119],
                        [58, 180],
                        [80, 185],
                        [104, 190],
                        [134, 192],
                        [185, 204]], # 224
            'ava': []
        }
    },

}