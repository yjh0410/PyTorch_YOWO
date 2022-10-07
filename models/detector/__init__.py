import torch
from .yowo.yowo import YOWO


# build YOWO detector
def build_model(args,
                d_cfg,
                m_cfg, 
                device, 
                num_classes=80, 
                trainable=False,
                eval_mode=False,
                resume=None):
    print('==============================')
    print('Build {} ...'.format(args.version.upper()))

    # Basic config
    if trainable:
        img_size = d_cfg['train_size']
        conf_thresh = d_cfg['conf_thresh_val']
        nms_thresh = d_cfg['nms_thresh_val']
    else:
        img_size = d_cfg['test_size']
        if eval_mode:
            conf_thresh = d_cfg['conf_thresh_val']
            nms_thresh = d_cfg['nms_thresh_val']
        else:
            conf_thresh = d_cfg['conf_thresh']
            nms_thresh = d_cfg['nms_thresh']

    # build YOWO
    model = YOWO(
        cfg=m_cfg,
        device=device,
        anchor_size=m_cfg['anchor_size'][args.dataset],
        img_size=img_size,
        len_clip=d_cfg['len_clip'],
        num_classes=num_classes,
        conf_thresh=conf_thresh,
        nms_thresh=nms_thresh,
        topk=args.topk,
        trainable=trainable,
        multi_hot=d_cfg['multi_hot']
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
