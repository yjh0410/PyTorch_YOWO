import cv2
import numpy as np


def vis_targets(video_clips, targets):
    """
        video_clips: Lits[Tensor] -> [Tensor[B, C, H, W], ..., Tensor[B, C, H, W]]
        targets: List[List] -> [List[B, N, 6], ..., List[B, N, 6]]
    """
    pixel_mean=(123.675, 116.28, 103.53)
    pixel_std=(58.395, 57.12, 57.375)
    len_clip = len(video_clips)
    batch_size = video_clips[0].shape[0]
    for batch_index in range(batch_size):
        for fid in range(len_clip):
            frame = video_clips[fid][batch_index] # [C, H, W]
            frame = frame.permute(1, 2, 0).cpu().numpy()
            # denormalize
            frame = frame * pixel_std + pixel_mean
            # To BGR
            frame = frame[:, :, (2, 1, 0)].astype(np.uint8)
            frame = frame.copy()

            target = targets[fid][batch_index]
            for tgt in target:
                x1, y1, x2, y2, label, fid = tgt
                # draw bbox
                cv2.rectangle(frame,
                                (int(x1), int(y1)),
                                (int(x2), int(y2)),
                                (255, 0, 0), 2)
            cv2.imshow('groundtruth', frame)
            cv2.waitKey(0)


def plot_bbox_labels(img, bbox, label=None, cls_color=None, text_scale=0.4):
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
    # plot bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), cls_color, 2)
    
    if label is not None:
        # plot title bbox
        cv2.rectangle(img, (x1, y1-t_size[1]), (int(x1 + t_size[0] * text_scale), y1), cls_color, -1)
        # put the test on the title bbox
        cv2.putText(img, label, (int(x1), int(y1 - 5)), 0, text_scale, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    return img


def vis_detection(frame, scores, labels, bboxes, vis_thresh, class_names, class_colors):
    ts = 0.4
    for i, bbox in enumerate(bboxes):
        if scores[i] > vis_thresh:
            label = int(labels[i])
            cls_color = class_colors[label]
                
            if len(class_names) > 1:
                mess = '%s: %.2f' % (class_names[label], scores[i])
            else:
                cls_color = [255, 0, 0]
                mess = None
                # visualize bbox
            frame = plot_bbox_labels(frame, bbox, mess, cls_color, text_scale=ts)

    return frame
        