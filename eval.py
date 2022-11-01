import argparse
import torch

from evaluator.ucf_jhmdb_evaluator import UCF_JHMDB_Evaluator
from evaluator.ava_evaluator import AVA_Evaluator

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
    parser.add_argument('--save_path', default='./evaluator/eval_results/',
                        type=str, help='Trained state_dict file path to open')

    # dataset
    parser.add_argument('-d', '--dataset', default='ucf24',
                        help='ucf24, jhmdb, ava_v2.2.')

    # eval
    parser.add_argument('--cal_frame_mAP', action='store_true', default=False, 
                        help='calculate frame mAP.')
    parser.add_argument('--cal_video_mAP', action='store_true', default=False, 
                        help='calculate video mAP.')

    # model
    parser.add_argument('-v', '--version', default='yowo', type=str, choices=['yowo', 'yowo_nano'],
                        help='build YOWO')
    parser.add_argument('--weight', default=None,
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--topk', default=40, type=int,
                        help='NMS threshold')

    return parser.parse_args()


def ucf_jhmdb_eval(args, d_cfg, model, transform, collate_fn):
    if args.cal_frame_mAP:
        # Frame mAP evaluator
        evaluator = UCF_JHMDB_Evaluator(
            data_root=d_cfg['data_root'],
            dataset=args.dataset,
            model_name=args.version,
            metric='fmap',
            img_size=d_cfg['test_size'],
            len_clip=d_cfg['len_clip'],
            batch_size=args.batch_size,
            conf_thresh=0.01,
            iou_thresh=0.5,
            transform=transform,
            collate_fn=collate_fn,
            gt_folder=d_cfg['gt_folder'],
            save_path=args.save_path
            )
        # evaluate
        evaluator.evaluate_frame_map(model, show_pr_curve=True)

    elif args.cal_video_mAP:
        # Video mAP evaluator
        evaluator = UCF_JHMDB_Evaluator(
            data_root=d_cfg['data_root'],
            dataset=args.dataset,
            model_name=args.version,
            metric='vmap',
            img_size=d_cfg['test_size'],
            len_clip=d_cfg['len_clip'],
            batch_size=args.batch_size,
            conf_thresh=0.01,
            iou_thresh=0.5,
            transform=transform,
            collate_fn=collate_fn,
            gt_folder=d_cfg['gt_folder'],
            save_path=args.save_path
            )
        # evaluate
        evaluator.evaluate_video_map(model)


def ava_eval(args, d_cfg, model, transform, collate_fn):
    evaluator = AVA_Evaluator(
        d_cfg=d_cfg,
        img_size=d_cfg['test_size'],
        len_clip=d_cfg['len_clip'],
        sampling_rate=d_cfg['sampling_rate'],
        batch_size=args.batch_size,
        transform=transform,
        collate_fn=collate_fn,
        full_test_on_val=False,
        version='v2.2')

    mAP = evaluator.evaluate_frame_map(model)


if __name__ == '__main__':
    args = parse_args()
    # dataset
    if args.dataset == 'ucf24':
        num_classes = 24

    elif args.dataset == 'jhmdb':
        num_classes = 21

    elif args.dataset == 'ava_v2.2':
        num_classes = 80

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
        trainable=False,
        eval_mode=True
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
    elif args.dataset == 'ava_v2.2':
        ava_eval(
            args=args,
            d_cfg=d_cfg,
            model=model,
            transform=basetransform,
            collate_fn=CollateFunc()
            )
