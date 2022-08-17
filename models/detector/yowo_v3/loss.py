import torch
import torch.nn as nn
from .matcher import Yolov3Matcher
from utils.box_ops import get_ious
from utils.misc import Softmax_FocalLoss
from utils.vis_tools import vis_targets



class Criterion(object):
    def __init__(self, 
                 cfg, 
                 device,
                 anchor_size,
                 num_anchors,
                 num_classes,
                 loss_obj_weight=5.0,
                 loss_noobj_weight=1.0,
                 loss_cls_weight=1.0, 
                 loss_reg_weight=1.0):
        self.cfg = cfg
        self.device = device
        self.anchor_size = anchor_size
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.loss_obj_weight = loss_obj_weight
        self.loss_noobj_weight = loss_noobj_weight
        self.loss_cls_weight = loss_cls_weight
        self.loss_reg_weight = loss_reg_weight

        # Matcher
        self.matcher = Yolov3Matcher(
            num_classes=num_classes,
            num_anchors=num_anchors,
            anchor_size=anchor_size,
            iou_thresh=cfg['ignore_thresh']
            )
        
        # Loss
        self.conf_loss = nn.MSELoss(reduction='none')
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
            targets: List[Dict] -> [{'boxes': (Tensor) [N, 4],
                                     'labels': (Tensor) [N,]}, 
                                     ...],
            video_clips: (Tensor) -> [B, C, T, H, W]
        """
        if vis_data:
            # To DO: 
            # vis video clip and targets
            vis_targets(video_clips, targets)

        # target of key-frame
        batch_size = outputs['conf_pred'][0].shape[0]
        device = outputs['conf_pred'][0].device
        fpn_strides = outputs['strides']

        # Matcher for this frame
        (
            gt_conf, 
            gt_cls, 
            gt_bboxes
            ) = self.matcher(img_size=outputs['img_size'], 
                             fpn_strides=fpn_strides, 
                             targets=targets)

        pred_conf = torch.cat(outputs['conf_pred'], dim=1).view(-1)                  # [BM,]
        pred_cls = torch.cat(outputs['cls_pred'], dim=1).view(-1, self.num_classes)  # [BM, C]
        pred_box = torch.cat(outputs['box_pred'], dim=1).view(-1, 4)                 # [BM, 4]
        
        gt_conf = gt_conf.flatten().to(device).float()        # [BM,]
        gt_cls = gt_cls.flatten().to(device).long()           # [BM,]
        gt_bboxes = gt_bboxes.view(-1, 4).to(device).float()  # [BM, 4]

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
    

if __name__ == "__main__":
    pass