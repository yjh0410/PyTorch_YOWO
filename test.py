import argparse
import cv2
import os
import time
import numpy as np
import torch

from dataset.ucf_jhmdb import UCF_JHMDB_Dataset
from dataset.transforms import BaseTransform

from utils.misc import load_weight
from utils.box_ops import rescale_bboxes
from utils.vis_tools import vis_detection

from config import build_dataset_config, build_model_config
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
    parser.add_argument('--save', action='store_true', default=False, 
                        help='save detection results.')
    parser.add_argument('--save_folder', default='det_results/', type=str,
                        help='Dir to save results')
    parser.add_argument('-vs', '--vis_thresh', default=0.35, type=float,
                        help='threshold for visualization')

    # model
    parser.add_argument('-v', '--version', default='yowo-d19', type=str,
                        help='build yowo')
    parser.add_argument('--weight', default=None,
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--topk', default=40, type=int,
                        help='NMS threshold')

    # dataset
    parser.add_argument('--root', default='/mnt/share/ssd2/dataset',
                        help='data root')
    parser.add_argument('-d', '--dataset', default='ucf24',
                        help='ucf24, ava.')

    return parser.parse_args()


@torch.no_grad()
def inference(args, model, device, dataset, class_names=None, class_colors=None):
    # path to save 
    if args.save:
        save_path = os.path.join(
            args.save_folder, args.dataset, 
            args.version, 'video_clips')
        os.makedirs(save_path, exist_ok=True)

    # inference
    for index in range(0, len(dataset)):
        print('Video clip {:d}/{:d}....'.format(index+1, len(dataset)))
        frame_id, video_clip, target = dataset[index]

        orig_size = target['orig_size']  # width, height

        # prepare
        video_clip = video_clip.unsqueeze(0).to(device) # [B, 3, T, H, W], B=1

        t0 = time.time()
        # inference
        batch_scores, batch_labels, batch_bboxes = model(video_clip)
        print("inference time ", time.time() - t0, "s")

        # batch size = 1
        scores = batch_scores[0]
        labels = batch_labels[0]
        bboxes = batch_bboxes[0]
        
        # rescale
        bboxes = rescale_bboxes(bboxes, orig_size)

        # vis results of key-frame
        key_frame = video_clip[0, :, -1, :, :]
        key_frame = (key_frame * 255).permute(1, 2, 0)
        key_frame = key_frame.cpu().numpy().astype(np.uint8)
        key_frame = key_frame.copy()[..., (2, 1, 0)]  # to BGR
        key_frame = cv2.resize(key_frame, orig_size)

        vis_results = vis_detection(
            frame=key_frame,
            scores=scores,
            labels=labels,
            bboxes=bboxes,
            vis_thresh=args.vis_thresh,
            class_names=class_names,
            class_colors=class_colors
            )

        if args.show:
            cv2.imshow('key-frame detection', vis_results)
            cv2.waitKey(0)

        if args.save:
            # save result
            cv2.imwrite(os.path.join(save_path,
            '{:0>5}.jpg'.format(index)), vis_results)
        

if __name__ == '__main__':
    args = parse_args()
    # cuda
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # config
    d_cfg = build_dataset_config(args)
    m_cfg = build_model_config(args)

    # transform
    basetransform = BaseTransform(img_size=m_cfg['test_size'])

    # dataset
    if args.dataset in ['ucf24', 'jhmdb21']:
        dataset = UCF_JHMDB_Dataset(
            data_root=d_cfg['data_root'],
            dataset=args.dataset,
            img_size=m_cfg['test_size'],
            transform=basetransform,
            is_train=False,
            len_clip=d_cfg['len_clip'],
            sampling_rate=d_cfg['sampling_rate']
            )
        class_names = d_cfg['label_map']
        num_classes = dataset.num_classes

    elif args.dataset == 'ava':
        dataset = None
        class_names = None
        num_classes = None

    else:
        print('unknow dataset !! Only support voc and coco !!')
        exit(0)

    np.random.seed(100)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(num_classes)]

    # build model
    model = build_model(
        args=args,
        d_cfg=d_cfg,
        m_cfg=m_cfg,
        device=device, 
        num_classes=num_classes, 
        trainable=False
        )

    # load trained weight
    model = load_weight(model=model, path_to_ckpt=args.weight)

    # to eval
    model = model.to(device).eval()

    # run
    inference(
        args=args,
        model=model,
        device=device,
        dataset=dataset,
        class_names=class_names,
        class_colors=class_colors
        )
