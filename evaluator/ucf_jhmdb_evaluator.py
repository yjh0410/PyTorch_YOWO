import os
import torch

from dataset.ucf_jhmdb import UCF_JHMDB_Dataset
from utils.box_ops import rescale_bboxes
from utils.box_ops import rescale_bboxes

from .cal_mAP import get_mAP
from .utils import bbox_iou


class UCF_JHMDB_Evaluator(object):
    def __init__(self,
                 device,
                 data_root=None,
                 dataset='ucf24',
                 img_size=224,
                 len_clip=1,
                 batch_size=1,
                 iou_thresh=0.5,
                 transform=None,
                 collate_fn=None,
                 cal_mAP=False,
                 gt_folder=None,
                 dt_folder=None,
                 save_path=None):
        self.device = device
        self.data_root = data_root
        self.dataset = dataset
        self.img_size = img_size
        self.len_clip = len_clip
        self.conf_thresh = 0.1
        self.iou_thresh = iou_thresh
        self.cal_mAP = cal_mAP

        self.dt_folder = dt_folder
        self.save_path = save_path
        self.gt_folder = gt_folder

        # dataset
        self.testset = UCF_JHMDB_Dataset(
            data_root=data_root,
            dataset=dataset,
            img_size=img_size,
            transform=transform,
            is_train=False,
            len_clip=len_clip,
            sampling_rate=1)
        self.num_classes = self.testset.num_classes

        # dataloader
        self.testloader = torch.utils.data.DataLoader(
            dataset=self.testset, 
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn, 
            num_workers=4,
            drop_last=False,
            pin_memory=True
            )
    

    def evaluate(self, model, epoch=1):
        # number of groundtruth
        total_num_gts = 0
        proposals   = 0.0
        correct     = 0.0
        fscore = 0.0

        correct_classification = 0.0
        total_detected = 0.0
        eps = 1e-5

        epoch_size = len(self.testloader)

        for iter_i, (batch_frame_id, batch_video_clip, batch_target) in enumerate(self.testloader):
            # to device
            batch_video_clip = batch_video_clip.to(self.device)

            with torch.no_grad():
                # inference
                batch_scores, batch_labels, batch_bboxes = model(batch_video_clip)

                # process batch
                for bi in range(len(batch_scores)):
                    frame_id = batch_frame_id[bi]
                    scores = batch_scores[bi]
                    labels = batch_labels[bi]
                    bboxes = batch_bboxes[bi]
                    target = batch_target[bi]

                    # rescale bbox
                    orig_size = target['orig_size']
                    bboxes = rescale_bboxes(bboxes, orig_size)

                    if not os.path.exists('results'):
                        os.mkdir('results')

                    if self.dataset == 'ucf24':
                        detection_path = os.path.join('results', 'ucf_detections', 'detections_' + str(epoch), frame_id)
                        current_dir = os.path.join('results', 'ucf_detections', 'detections_' + str(epoch))
                        if not os.path.exists('results/ucf_detections'):
                            os.mkdir('results/ucf_detections')
                        if not os.path.exists(current_dir):
                            os.mkdir(current_dir)
                    else:
                        detection_path = os.path.join('results', 'jhmdb_detections', 'detections_' + str(epoch), frame_id)
                        current_dir = os.path.join('results', 'jhmdb_detections', 'detections_' + str(epoch))
                        if not os.path.exists('results/jhmdb_detections'):
                            os.mkdir('results/jhmdb_detections')
                        if not os.path.exists(current_dir):
                            os.mkdir(current_dir)

                    with open(detection_path, 'w+') as f_detect:
                        for score, label, bbox in zip(scores, labels, bboxes):
                            x1 = round(bbox[0])
                            y1 = round(bbox[1])
                            x2 = round(bbox[2])
                            y2 = round(bbox[3])
                            cls_id = int(label) + 1

                            f_detect.write(
                                str(cls_id) + ' ' + str(score) + ' ' \
                                    + str(x1) + ' ' + str(y1) + ' ' + str(x2) + ' ' + str(y2) + '\n')

                    tgt_bboxes = target['boxes'].numpy()
                    tgt_labels = target['labels'].numpy()
                    ow, oh = target['orig_size']
                    num_gts = tgt_bboxes.shape[0]

                    # count number of total groundtruth
                    total_num_gts += num_gts

                    pred_list = [] # LIST OF CONFIDENT BOX INDICES
                    for i in range(len(scores)):
                        score = scores[i]
                        if score > self.conf_thresh:
                            proposals += 1
                            pred_list.append(i)

                    for i in range(num_gts):
                        tgt_bbox = tgt_bboxes[i]
                        tgt_label = tgt_labels[i]
                        # rescale groundtruth bbox
                        tgt_bbox[[0, 2]] *= ow
                        tgt_bbox[[1, 3]] *= oh

                        tgt_x1, tgt_y1, tgt_x2, tgt_y2 = tgt_bbox
                        box_gt = [tgt_x1, tgt_y1, tgt_x2, tgt_y2, 1.0, 1.0, tgt_label]
                        best_iou = 0
                        best_j = -1

                        for j in pred_list: # ITERATE THROUGH ONLY CONFIDENT BOXES
                            iou = bbox_iou(box_gt, bboxes[j], x1y1x2y2=True)
                            if iou > best_iou:
                                best_j = j
                                best_iou = iou

                        if best_iou > self.iou_thresh:
                            total_detected += 1
                            if int(labels[best_j]) == int(tgt_label):
                                correct_classification += 1

                        if best_iou > self.iou_thresh and int(labels[best_j]) == int(tgt_label):
                            correct += 1

                precision = 1.0 * correct / (proposals + eps)
                recall = 1.0 * correct / (total_num_gts + eps)
                fscore = 2.0 * precision * recall / (precision + recall + eps)

                if iter_i % 100 == 0:
                    log_info = "[%d / %d] precision: %f, recall: %f, fscore: %f" % (iter_i, epoch_size, precision, recall, fscore)
                    print(log_info, flush=True)

        classification_accuracy = 1.0 * correct_classification / (total_detected + eps)
        locolization_recall = 1.0 * total_detected / (total_num_gts + eps)

        print("Classification accuracy: %.3f" % classification_accuracy)
        print("Locolization recall: %.3f" % locolization_recall)

        # frame mAP
        if self.cal_mAP:
            print('calculating Frame mAP ...')
            if self.dt_folder is None:
                result_path = current_dir
            else:
                result_path = self.dt_folder

            metric_list = get_mAP(self.gt_folder, result_path, self.iou_thresh ,self.save_path)
            print(metric_list)


        return classification_accuracy, locolization_recall

        

if __name__ == "__main__":
    pass
