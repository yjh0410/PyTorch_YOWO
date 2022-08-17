import torch

from .yowo.yowo import YOWO
from .yowo_v2.yowo_v2 import YOWOv2
from .yowo_v3.yowo_v3 import YOWOv3


# build YOWO detector
def build_model(args,
                d_cfg,
                m_cfg, 
                device, 
                num_classes=80, 
                trainable=False,
                resume=None):
    print('==============================')
    print('Build {} ...'.format(args.version.upper()))

    # Basic config
    if trainable:
        img_size = d_cfg['train_size']
    else:
        img_size = d_cfg['test_size']

    # build YOWO
    if args.version == 'yowo':
        model = YOWO(
            cfg=m_cfg,
            device=device,
            anchor_size=m_cfg['anchor_size'][args.dataset],
            img_size=img_size,
            len_clip=d_cfg['len_clip'],
            num_classes=num_classes,
            conf_thresh=m_cfg['conf_thresh'],
            nms_thresh=m_cfg['nms_thresh'],
            topk=args.topk,
            trainable=trainable,
            cls_prob=d_cfg['cls_prob']
            )

    elif args.version == 'yowo_v2':
        model = YOWOv2(
            cfg=m_cfg,
            device=device,
            anchor_size=m_cfg['anchor_size'][args.dataset],
            img_size=img_size,
            len_clip=d_cfg['len_clip'],
            num_classes=num_classes,
            conf_thresh=m_cfg['conf_thresh'],
            nms_thresh=m_cfg['nms_thresh'],
            topk=args.topk,
            trainable=trainable,
            cls_prob=d_cfg['cls_prob']
            )
            
    elif args.version == 'yowo_v3':
        model = YOWOv3(
            cfg=m_cfg,
            device=device,
            anchor_size=m_cfg['anchor_size'][args.dataset],
            img_size=img_size,
            len_clip=d_cfg['len_clip'],
            num_classes=num_classes,
            conf_thresh=m_cfg['conf_thresh'],
            nms_thresh=m_cfg['nms_thresh'],
            topk=args.topk,
            trainable=trainable,
            cls_prob=d_cfg['cls_prob']
            )

    # Freeze backbone
    if d_cfg['freeze_backbone_2d']:
        print('Freeze 2D Backbone ...')
        for m in model.backbone_2d.parameters():
            m.requires_grad = False
    if d_cfg['freeze_backbone_3d']:
        print('Freeze 3D Backbone ...')
        for m in model.backbone_3d.parameters():
            m.requires_grad = False
            
    # keep training       
    if resume is not None:
        print('keep training: ', resume)
        checkpoint = torch.load(resume, map_location='cpu')
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("model")
        model.load_state_dict(checkpoint_state_dict)
                        
    return model
