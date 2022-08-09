import argparse
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
    parser.add_argument('-mt', '--metrics', default=['frame_map', 'video_map'], type=str,
                        help='evaluation metrics')
    parser.add_argument('--save_dir', default='inference_results/',
                        type=str, help='Trained state_dict file path to open')

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


def ucf_jhmdb_eval(device, args, d_cfg, model, transform, collate_fn):
    evaluator = UCF_JHMDB_Evaluator(
        device=device,
        dataset=args.dataset,
        batch_size=args.batch_size,
        data_root=d_cfg['data_root'],
        img_size=d_cfg['test_size'],
        len_clip=d_cfg['len_clip'],
        conf_thresh=0.1,
        iou_thresh=0.5,
        transform=transform,
        collate_fn=collate_fn)

    cls_accu, loc_recall = evaluator.evaluate(model)



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
    basetransform = BaseTransform(img_size=d_cfg['test_size'])

    # path to save inference results
    save_path = os.path.join(args.save_dir, args.dataset)

    # run
    if args.dataset in ['ucf24', 'jhmdb21']:
        ucf_jhmdb_eval(
            device=device,
            args=args,
            d_cfg=d_cfg,
            model=model,
            transform=basetransform,
            collate_fn=CollateFunc()
            )
