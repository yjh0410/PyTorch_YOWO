# This is a frame-level model which is set as the Baseline
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...backbone import build_backbone_2d
from ...backbone import build_backbone_3d
from .encoder import ChannelEncoder, SpatialEncoder
from .loss import Criterion



class YOWOv3(nn.Module):
    def __init__(self, 
                 cfg,
                 device,
                 img_size,
                 anchor_size,
                 len_clip = 16,
                 num_classes = 20, 
                 conf_thresh = 0.05,
                 nms_thresh = 0.6,
                 topk = 1000,
                 trainable = False):
        super(YOWOv3, self).__init__()
        self.cfg = cfg
        self.device = device
        self.img_size = img_size
        self.stride = cfg['stride']
        self.len_clip = len_clip
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.topk = topk
        ## anchor config
        self.num_levels = len(cfg['stride'])
        self.num_anchors = len(anchor_size) // len(cfg['stride'])
        self.anchor_size = torch.as_tensor(
            anchor_size
            ).view(self.num_levels, self.num_anchors, 2) # [S, KA, 2]

        # ------------------ Anchor Box --------------------
        # [M, 4]
        self.anchor_boxes = self.generate_anchors(img_size)

        # ------------------ Network ---------------------
        # 2D backbone
        self.backbone_2d, bk_dim_2d = build_backbone_2d(
            model_name=cfg['backbone_2d'], 
            pretrained=cfg['pretrained_2d'] and trainable
            )
            
        # 3D backbone
        self.backbone_3d, bk_dim_3d = build_backbone_3d(
            model_name=cfg['backbone_3d'],
            pretrained=cfg['pretrained_2d'] and trainable
        )

        # spatial encoder
        self.spatial_encoders = nn.ModuleList([
            ChannelEncoder(
                in_dim=bk_dim_2d[i],
                out_dim=bk_dim_2d[i],
                act_type=cfg['head_act'],
                norm_type=cfg['head_norm']
                )
            for i in range(len(self.stride))
        ])

        # channel encoder
        self.channel_encoders = nn.ModuleList([
            ChannelEncoder(
                in_dim=bk_dim_2d[i] + bk_dim_3d,
                out_dim=cfg['head_dim'][i],
                act_type=cfg['head_act'],
                norm_type=cfg['head_norm']
                )
            for i in range(len(self.stride))
        ])
        
        # output
        self.preds = nn.ModuleList([
            nn.Conv2d(cfg['head_dim'][i], self.num_anchors * (1 + num_classes + 4), kernel_size=1)
            for i in range(len(self.stride))
        ])


        if trainable:
            # init bias
            self._init_bias()

        # ------------------ Criterion ---------------------
        if self.trainable:
            self.criterion = Criterion(
                cfg=cfg,
                device=device,
                num_anchors=self.num_anchors,
                num_classes=self.num_classes,
                anchor_size=anchor_size,
                loss_obj_weight=cfg['loss_obj_weight'],
                loss_noobj_weight=cfg['loss_noobj_weight'],
                loss_cls_weight=cfg['loss_cls_weight'],
                loss_reg_weight=cfg['loss_reg_weight']
                )


    def _init_bias(self):  
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        # init bias
        for pred in self.preds:
            nn.init.constant_(pred.bias[..., :self.num_anchors], bias_value)
            nn.init.constant_(pred.bias[..., self.num_anchors:self.num_anchors*(1 + self.num_classes)], bias_value)


    def generate_anchors(self, img_size):
        """
            fmp_size: (List) [H, W]
        """
        all_anchor_boxes = []
        for level, stride in enumerate(self.stride):
            fmp_h = fmp_w = img_size // stride
            # [KA, 2]
            anchor_size = self.anchor_size[level]

            # generate grid cells
            anchor_y, anchor_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
            anchor_xy = torch.stack([anchor_x, anchor_y], dim=-1).float().view(-1, 2)
            # [HW, 2] -> [HW, KA, 2] -> [M, 2]
            anchor_xy = anchor_xy.unsqueeze(1).repeat(1, self.num_anchors, 1)
            anchor_xy = anchor_xy.view(-1, 2).to(self.device)
            anchor_xy *= stride

            # [KA, 2] -> [1, KA, 2] -> [HW, KA, 2] -> [M, 2]
            anchor_wh = anchor_size.unsqueeze(0).repeat(fmp_h*fmp_w, 1, 1)
            anchor_wh = anchor_wh.view(-1, 2).to(self.device)

            # [HW, KA, 4] -> [M, 4]
            anchor_boxes = torch.cat([anchor_xy, anchor_wh], dim=-1)
            anchor_boxes = anchor_boxes.view(-1, 4).to(self.device)

            all_anchor_boxes.append(anchor_boxes)

        return all_anchor_boxes
        

    def decode_boxes(self, level, anchors, pred_reg):
        """
            anchors:  (List[Tensor]) [M, 4]
            pred_reg: (List[Tensor]) [M, 4]
        """
        pred_ctr_delta = pred_reg[..., :2].sigmoid() * self.stride[level]
        pred_ctr = anchors[..., :2] + pred_ctr_delta
        pred_wh = anchors[..., 2:] * pred_reg[..., 2:].exp()
        
        pred_x1y1 = pred_ctr - pred_wh * 0.5
        pred_x2y2 = pred_ctr + pred_wh * 0.5
        pred_box = torch.cat([pred_x1y1, pred_x2y2], dim=-1)

        return pred_box


    def nms(self, dets, scores):
        """"Pure Python NMS."""
        x1 = dets[:, 0]  #xmin
        y1 = dets[:, 1]  #ymin
        x2 = dets[:, 2]  #xmax
        y2 = dets[:, 3]  #ymax

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            # compute iou
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            # intersection
            w = np.maximum(1e-10, xx2 - xx1)
            h = np.maximum(1e-10, yy2 - yy1)
            inter = w * h

            # union
            union = areas[i] + areas[order[1:]] - inter

            # iou
            iou = inter / np.clip(union, a_min=1e-10, a_max=np.inf)

            #nms thresh
            inds = np.where(iou <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep


    def post_process(self, scores, labels, bboxes):
        # threshold
        keep = np.where(scores >= self.conf_thresh)
        scores = scores[keep]
        labels = labels[keep]
        bboxes = bboxes[keep]

        # nms
        keep = np.zeros(len(bboxes), dtype=np.int)
        for i in range(self.num_classes):
            inds = np.where(labels == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        scores = scores[keep]
        labels = labels[keep]
        bboxes = bboxes[keep]

        return scores, labels, bboxes
    

    @torch.no_grad()
    def inference(self, video_clips):
        """
        Input:
            video_clips: (Tensor) -> [B, 3, T, H, W].
        """                        
        key_frame = video_clips[:, :, -1, :, :]
        # backbone
        feats_2d = self.backbone_2d(key_frame)              # [[B, C1, H, W], ...]
        feat_3d = self.backbone_3d(video_clips).squeeze(2)  # [B, C2, H, W]

        conf_pred_list = []
        cls_pred_list = []
        box_pred_list = []
        # head
        for level in range(self.num_levels):
            feat_2d = feats_2d[level]

            # upsample
            if level < self.num_levels - 1:
                feat_3d_ = F.interpolate(feat_3d, scale_factor=2**(self.num_levels - 1 - level))
            else:
                feat_3d_ = feat_3d

            # spatial encoder
            feat_2d = self.spatial_encoders[level](feat_2d)
            
            # channel encoder
            feat = torch.cat([feat_2d, feat_3d_], dim=1)
            feat = self.channel_encoders[level](feat)
            
            # pred
            pred = self.preds[level](feat)

            B, K, C = pred.size(0), self.num_anchors, self.num_classes
            # [B, K*C, H, W] -> [B, H, W, K*C] -> [B, M, C], M = HWK
            conf_pred = pred[:, :K, :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, 1)
            cls_pred = pred[:, K:K*(1 + C), :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, C)
            reg_pred = pred[:, K*(1 + C):, :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, 4)
            # decode box
            anchor_boxes = self.anchor_boxes[level]
            box_pred = self.decode_boxes(level, anchor_boxes[None], reg_pred)

            conf_pred_list.append(conf_pred)
            cls_pred_list.append(cls_pred)
            box_pred_list.append(box_pred)
        
        conf_pred = torch.cat(conf_pred_list, dim=1)  # [B, M, 1]
        cls_pred = torch.cat(cls_pred_list, dim=1)    # [B, M, C]
        box_pred = torch.cat(box_pred_list, dim=1)    # [B, M, 4]

        batch_scores = []
        batch_labels = []
        batch_bboxes = []
        for batch_idx in range(conf_pred.size(0)):
            # [B, M, C] -> [M, C]
            cur_conf_pred = conf_pred[batch_idx]
            cur_cls_pred = cls_pred[batch_idx]
            cur_box_pred = box_pred[batch_idx]
                        
            # scores
            scores, labels = torch.max(torch.sigmoid(cur_conf_pred) * torch.softmax(cur_cls_pred, dim=-1), dim=-1)

            # topk
            if scores.shape[0] > self.topk:
                scores, indices = torch.topk(scores, self.topk)
                labels = labels[indices]
                cur_box_pred = cur_box_pred[indices]

            # normalize box
            bboxes = torch.clamp(cur_box_pred / self.img_size, 0., 1.)
            
            # to cpu
            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()
            bboxes = bboxes.cpu().numpy()

            # post-process
            scores, labels, bboxes = self.post_process(scores, labels, bboxes)

            batch_scores.append(scores)
            batch_labels.append(labels)
            batch_bboxes.append(bboxes)

        return batch_scores, batch_labels, batch_bboxes


    def forward(self, video_clips, targets=None):
        """
        Input:
            video_clips: (Tensor) -> [B, 3, T, H, W].
            targets: (List) -> [[x1, y1, x2, y2, label, frame_id], ...]
        """                        
        if not self.trainable:
            return self.inference(video_clips)
        else:
            key_frame = video_clips[:, :, -1, :, :]
            # backbone
            feats_2d = self.backbone_2d(key_frame)              # [[B, C1, H, W], ...]
            feat_3d = self.backbone_3d(video_clips).squeeze(2)  # [B, C2, H, W]

            all_conf_preds = []
            all_cls_preds = []
            all_box_preds = []
            # head
            for level in range(self.num_levels):
                feat_2d = feats_2d[level]

                # upsample
                if level < self.num_levels - 1:
                    feat_3d_ = F.interpolate(feat_3d, scale_factor=2**(self.num_levels - 1 - level))
                else:
                    feat_3d_ = feat_3d

                # spatial encoder
                feat_2d = self.spatial_encoders[level](feat_2d)

                # channel encoder
                feat = torch.cat([feat_2d, feat_3d_], dim=1)
                feat = self.channel_encoders[level](feat)
                
                # pred
                pred = self.preds[level](feat)

                B, K, C = pred.size(0), self.num_anchors, self.num_classes
                # [B, K*C, H, W] -> [B, H, W, K*C] -> [B, M, C], M = HWK
                conf_pred = pred[:, :K, :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, 1)
                cls_pred = pred[:, K:K*(1 + C), :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, C)
                reg_pred = pred[:, K*(1 + C):, :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, 4)

                # decode box
                anchor_boxes = self.anchor_boxes[level]
                box_pred = self.decode_boxes(level, anchor_boxes[None], reg_pred)

                all_conf_preds.append(conf_pred)
                all_cls_preds.append(cls_pred)
                all_box_preds.append(box_pred)

            outputs = {"conf_pred": all_conf_preds,
                       "cls_pred": all_cls_preds,
                       "box_pred": all_box_preds,
                       "img_size": self.img_size,
                       "strides": self.stride}

            # loss
            loss_dict = self.criterion(
                outputs=outputs, 
                targets=targets, 
                video_clips=video_clips
                )

            return loss_dict
