from math import gamma
import torch
import torch.nn as nn
from .matcher import YoloMatcher
from utils.box_ops import get_ious
from utils.misc import AVA_FocalLoss, Softmax_FocalLoss
from utils.vis_tools import vis_targets



class Criterion(object):
    def __init__(self, 
                 cfg, 
                 device,
                 anchor_size,
                 num_anchors,
                 num_classes,
                 multi_hot=False,
                 loss_obj_weight=5.0,
                 loss_noobj_weight=1.0,
                 loss_cls_weight=1.0, 
                 loss_reg_weight=1.0):
        self.cfg = cfg
        self.device = device
        self.anchor_size = anchor_size
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.multi_hot = multi_hot
        self.loss_obj_weight = loss_obj_weight
        self.loss_noobj_weight = loss_noobj_weight
        self.loss_cls_weight = loss_cls_weight
        self.loss_reg_weight = loss_reg_weight

        # Matcher
        self.matcher = YoloMatcher(
            num_classes=num_classes,
            num_anchors=num_anchors,
            anchor_size=anchor_size,
            iou_thresh=cfg['ignore_thresh'],
            multi_hot=multi_hot
            )
        
        # Loss
        self.conf_loss = nn.MSELoss(reduction='none')
        if multi_hot:
            self.cls_loss = AVA_FocalLoss(device, 0.5, num_classes, reduction='none')
        else:
            # self.cls_loss = nn.CrossEntropyLoss(reduction='none')
            self.cls_loss = Softmax_FocalLoss(num_classes=num_classes, gamma=2.0, reduction='none')


    def __call__(self, 
                 outputs, 
                 targets, 
                 video_clips=None, 
                 vis_data=False):
        """
            outputs['conf_pred']: [B, M, 1]
            outputs['cls_pred']: [B, M, C]
            outputs['box_pred]: [B, M, 4]
            outputs['stride']: Int -> stride of the model output
            anchor_size: (Tensor) [K, 2]
            targets: List[List] -> [List[B, N, 6], 
                                    ...,
                                    List[B, N, 6]],
            video_clips: Lits[Tensor] -> [Tensor[B, C, H, W], 
                                          ..., 
                                          Tensor[B, C, H, W]]
        """
        if vis_data:
            # To DO: 
            # vis video clip and targets
            vis_targets(video_clips, targets)

        # target of key-frame
        device = outputs['conf_pred'].device
        batch_size = outputs['conf_pred'].shape[0]

        # Matcher for this frame
        (
            gt_conf, 
            gt_cls, 
            gt_bboxes
            ) = self.matcher(img_size=outputs['img_size'], 
                             stride=outputs['stride'], 
                             targets=targets)

        pred_conf = outputs['conf_pred'].view(-1)                  # [BM,]
        pred_cls = outputs['cls_pred'].view(-1, self.num_classes)  # [BM, C]
        pred_box = outputs['box_pred'].view(-1, 4)                 # [BM, 4]
        
        gt_conf = gt_conf.flatten().to(device).float()                     # [BM,]
        gt_bboxes = gt_bboxes.view(-1, 4).to(device).float()               # [BM, 4]
        if self.multi_hot:
            gt_cls = gt_cls.view(-1, self.num_classes).to(device).long()  # [BM, C]
        else:
            gt_cls = gt_cls.flatten().to(device).long()                    # [BM,]

        # fore mask
        foreground_mask = (gt_conf > 0.)

        # box loss
        matched_pred_box = pred_box[foreground_mask]
        matched_tgt_box = gt_bboxes[foreground_mask]
        ious = get_ious(matched_pred_box,
                        matched_tgt_box,
                        box_mode="xyxy",
                        iou_type='giou')
        loss_box = (1.0 - ious).sum() / batch_size * self.loss_reg_weight

        # cls loss
        matched_pred_cls = pred_cls[foreground_mask]
        matched_tgt_cls = gt_cls[foreground_mask]
        loss_cls = self.cls_loss(matched_pred_cls, matched_tgt_cls)
        loss_cls = loss_cls.sum() / batch_size * self.loss_cls_weight

        # conf loss
        ## obj & noobj
        obj_mask = (gt_conf > 0.)
        noobj_mask = (gt_conf == 0.)
        ## gt conf with iou-aware
        gt_ious = torch.zeros_like(gt_conf)
        gt_ious[foreground_mask] = ious.clone().detach().clamp(0.)
        gt_conf = gt_conf * gt_ious
        ## weighted loss of conf
        loss = self.conf_loss(pred_conf.sigmoid(), gt_conf)
        loss_conf = loss * obj_mask * self.loss_obj_weight + \
                    loss * noobj_mask * self.loss_noobj_weight
        loss_conf = loss_conf.sum() / batch_size

        # total loss
        losses = loss_conf + loss_cls + loss_box

        
        loss_dict = dict(
                loss_conf = loss_conf,
                loss_cls = loss_cls,
                loss_box = loss_box,
                losses = losses
        )

        return loss_dict
    

def build_criterion(cfg, device, num_classes, anchor_size, multi_hot=False):
    criterion = Criterion(
        cfg=cfg,
        device=device,
        anchor_size=anchor_size,
        num_anchors=len(anchor_size),
        num_classes=num_classes,
        multi_hot=multi_hot,
        loss_obj_weight=cfg['loss_obj_weight'],
        loss_noobj_weight=cfg['loss_noobj_weight'],
        loss_cls_weight=cfg['loss_cls_weight'],
        loss_reg_weight=cfg['loss_reg_weight']
        )

    return criterion


if __name__ == "__main__":
    pass
