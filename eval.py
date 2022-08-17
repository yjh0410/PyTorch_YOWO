import argparse
from operator import gt
import os
import torch

from evaluator.ucf_jhmdb_evaluator import UCF_JHMDB_Evaluator

from dataset.transforms import BaseTransform

from utils.misc import load_weight, CollateFunc

from config import build_dataset_config, build_model_config
from models.detector import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='YOWO')

    # basic
    parser.add_argument('-bs', '--batch_size', default=8, type=int,
                        help='test batch size')
    parser.add_argument('-size', '--img_size', default=224, type=int,
                        help='the size of input frame')
    parser.add_argument('--cuda', action='store_true', default=False, 
                        help='use cuda.')
    parser.add_argument('--save_path', default='results/',
                        type=str, help='Trained state_dict file path to open')

    # dataset
    parser.add_argument('-d', '--dataset', default='ucf24',
                        help='ucf24, jhmdb, ava.')

    # eval
    parser.add_argument('--gt_folder', default='./evaluator/groundtruth_ucf_jhmdb/groundtruth_ucf/',
                        type=str, help='path to grouondtruth of ucf & jhmdb')
    parser.add_argument('--dt_folder', default=None,
                        type=str, help='path to detection dir')
    parser.add_argument('--cal_mAP', action='store_true', default=False, 
                        help='calculate_mAP.')
    parser.add_argument('--redo', action='store_true', default=False, 
                        help='re-make inference on testset.')

    # model
    parser.add_argument('-v', '--version', default='yowo', type=str,
                        help='build yowo')
    parser.add_argument('--weight', default=None,
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--topk', default=40, type=int,
                        help='NMS threshold')

    return parser.parse_args()


def ucf_jhmdb_eval(args, d_cfg, model, transform, collate_fn):
    evaluator = UCF_JHMDB_Evaluator(
        dataset=args.dataset,
        model_name=args.version,
        batch_size=args.batch_size,
        data_root=d_cfg['data_root'],
        img_size=d_cfg['test_size'],
        len_clip=d_cfg['len_clip'],
        conf_thresh=0.01,
        iou_thresh=0.5,
        transform=transform,
        collate_fn=collate_fn,
        redo=args.redo,
        gt_folder=args.gt_folder,
        dt_folder=args.dt_folder,
        save_path=args.save_path)

    if args.cal_mAP:
        evaluator.evaluate_frame_map(model, show_pr_curve=True)
    else:
        cls_accu, loc_recall = evaluator.evaluate_accu_recall(model)



if __name__ == '__main__':
    args = parse_args()
    # dataset
    if args.dataset == 'ucf24':
        num_classes = 24

    elif args.dataset == 'jhmdb':
        num_classes = 21
    
    else:
        print('unknow dataset.')
        exit(0)

    # cuda
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # config
    d_cfg = build_dataset_config(args)
    m_cfg = build_model_config(args)


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

    # transform
    basetransform = BaseTransform(
        img_size=d_cfg['test_size'],
        pixel_mean=d_cfg['pixel_mean'],
        pixel_std=d_cfg['pixel_std']
        )

    # run
    if args.dataset in ['ucf24', 'jhmdb21']:
        ucf_jhmdb_eval(
            args=args,
            d_cfg=d_cfg,
            model=model,
            transform=basetransform,
            collate_fn=CollateFunc()
            )
