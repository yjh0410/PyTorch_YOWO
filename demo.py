import argparse
import cv2
import os
import time
import numpy as np
import torch

from dataset.ucf24 import UCF24_CLASSES
from dataset.transforms import ValTransforms
from utils.misc import load_weight, rescale_bboxes, rescale_bboxes_list
from utils.vis_tools import vis_video_clip, vis_video_frame

from config import build_model_config
from models.detector import build_model



def parse_args():
    parser = argparse.ArgumentParser(description='YOWO')

    # basic
    parser.add_argument('-size', '--img_size', default=224, type=int,
                        help='the size of input frame')
    parser.add_argument('--show', action='store_true', default=False,
                        help='show the visulization results.')
    parser.add_argument('--cuda', action='store_true', default=False, 
                        help='use cuda.')
    parser.add_argument('--save_folder', default='det_results/', type=str,
                        help='Dir to save results')
    parser.add_argument('-vs', '--vis_thresh', default=0.35, type=float,
                        help='threshold for visualization')
    parser.add_argument('-path', '--path_to_video', default='data/demo/videos/', type=str,
                        help='path to video.')

    # model
    parser.add_argument('-v', '--version', default='yowo-d19', type=str,
                        help='build yowo')
    parser.add_argument('--weight', default='weight/',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--topk', default=40, type=int,
                        help='NMS threshold')

    # dataset
    parser.add_argument('--root', default='/mnt/share/ssd2/dataset',
                        help='data root')
    parser.add_argument('-d', '--dataset', default='coco',
                        help='coco, voc.')

    return parser.parse_args()
                    

def detect(args, model, device, transform=None, class_names=None):
    # path to save 
    save_path = os.path.join(args.save_folder, args.version, 'demo')
    os.makedirs(save_path, exist_ok=True)

    video = cv2.VideoCapture(args.path_to_video)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    save_size = (640, 480)
    save_name = os.path.join(save_path, 'detection.avi')
    fps = 15.0
    out = cv2.VideoWriter(save_name, fourcc, fps, save_size)
    frame_index = 0

    while(True):
        ret, frame = video.read()
        
        if ret:
            # prepare
            orig_h, orig_w = frame.shape[:2]
            orig_size = np.array([[orig_w, orig_h, orig_w, orig_h]])
            initialization = True
            init_video_clip = []
            scores_list = []
            labels_list = []
            bboxes_list = []

            if initialization:
                if frame_index < model.len_clip:
                    init_video_clip.append(frame)
                else:
                    # preprocess
                    xs, _ = transform(init_video_clip)

                    # to device
                    xs = [x.unsqueeze(0).to(device) for x in xs] 

                    # inference with an init video clip
                    init_scores, init_labels, init_bboxes = model(xs)

                    # rescale
                    init_bboxes = rescale_bboxes_list(init_bboxes, orig_size)

                    # store init predictions
                    scores_list.extend(init_scores)
                    labels_list.extend(init_labels)
                    bboxes_list.extend(init_bboxes)

                    initialization = False
                    del init_video_clip

                    # vis init detection
                    vis_results = vis_video_clip(
                        init_video_clip, scores_list, labels_list, bboxes_list, 
                        args.vis_thresh, class_names, splice=False
                    )

                    # write each frame to video
                    for frame in vis_results:
                        resized_frame = cv2.resize(frame, save_size)
                        out.write(resized_frame)
                        cv2.imshow('detection', frame)
                        cv2.waitKey(1)
                
                # count frame
                frame_index += 1
            else:
                # preprocess
                xs, _ = transform([frame])

                # to device
                xs = [x.unsqueeze(0).to(device) for x in xs] 

                # inference with the current frame
                t0 = time.time()
                cur_score, cur_label, cur_bboxes = model(xs[0])
                t1 = time.time()

                print('inference time: {:.3f}'.format(t1 - t0))
                
                # rescale
                cur_bboxes = rescale_bboxes(cur_bboxes, orig_size)

                # store current predictions
                scores_list.append(cur_score)
                labels_list.append(cur_label)
                bboxes_list.append(cur_bboxes)

                # vis current detection results
                vis_results = vis_video_frame(
                    frame, cur_score, cur_label, cur_bboxes, 
                    args.vis_thresh, class_names
                )

                # write cur frame to video
                resized_frame = cv2.resize(vis_results, save_size)
                out.write(resized_frame)
                cv2.imshow('detection', vis_results)
                cv2.waitKey(1)

                # count frame
                frame_index += 1

        else:
            break

    video.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    args = parse_args()

    np.random.seed(0)
    num_classes = 24
    class_names = UCF24_CLASSES
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(num_classes)]

    # cuda
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # config
    m_cfg = build_model_config(args)

    # build model
    model = build_model(args=args, 
                        cfg=m_cfg,
                        device=device, 
                        num_classes=num_classes, 
                        trainable=False)

    # load trained weight
    model = load_weight(device=device, 
                        model=model, 
                        path_to_ckpt=args.weight)

    # transform
    transform = ValTransforms(img_size=args.img_size,
                              pixel_mean=m_cfg['pixel_mean'],
                              pixel_std=m_cfg['pixel_std'],
                              format=m_cfg['format'])

    # run
    detect(args=args, net=model, device=device,
            transform=transform, class_names=class_names)
