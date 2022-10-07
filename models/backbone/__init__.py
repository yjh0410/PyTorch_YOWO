# import 2D backbone
from .backbone_2d.yolov2 import build_yolov2
from .backbone_2d.yolov3 import build_yolov3
# import 3D backbone
from .backbone_3d.resnet import build_resnet_3d
from .backbone_3d.resnext import build_resnext_3d
from .backbone_3d.shufflnetv2 import build_shufflenetv2_3d


def build_backbone_2d(cfg, pretrained=False):
    print('==============================')
    print('2D Backbone: {}'.format(cfg['backbone_2d'].upper()))
    print('--pretrained: {}'.format(pretrained))

    if cfg['backbone_2d'] == 'yolov2':
        model, feat_dim = build_yolov2(pretrained)

    elif cfg['backbone_2d'] == 'yolov3':
        model, feat_dim = build_yolov3(pretrained)

    else:
        print('Unknown 2D Backbone ...')
        exit()

    return model, feat_dim


def build_backbone_3d(cfg, pretrained=False):
    print('==============================')
    print('3D Backbone: {}'.format(cfg['backbone_3d'].upper()))
    print('--pretrained: {}'.format(pretrained))

    if 'resnet' in cfg['backbone_3d']:
        model, feat_dim = build_resnet_3d(
            model_name=cfg['backbone_3d'],
            pretrained=pretrained
            )
    elif 'resnext' in cfg['backbone_3d']:
        model, feat_dim = build_resnext_3d(
            model_name=cfg['backbone_3d'],
            pretrained=pretrained
            )
    elif 'shufflenetv2' in cfg['backbone_3d']:
        model, feat_dim = build_shufflenetv2_3d(
            model_size=cfg['model_size'],
            pretrained=pretrained
            )
    else:
        print('Unknown Backbone ...')
        exit()

    return model, feat_dim
