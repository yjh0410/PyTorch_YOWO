import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from copy import deepcopy
import math
import json

from dataset.ucf_jhmdb import UCF_JHMDB_Dataset
from dataset.ava import AVA_Dataset
from dataset.transforms import Augmentation, BaseTransform

from evaluator.ucf_jhmdb_evaluator import UCF_JHMDB_Evaluator
from evaluator.ava_evaluator import AVA_Evaluator


def build_dataset(d_cfg, args, is_train=False):
    """
        d_cfg: dataset config
    """
    # transform
    augmentation = Augmentation(
        img_size=d_cfg['train_size'],
        pixel_mean=d_cfg['pixel_mean'],
        pixel_std=d_cfg['pixel_std'],
        jitter=d_cfg['jitter'],
        hue=d_cfg['hue'],
        saturation=d_cfg['saturation'],
        exposure=d_cfg['exposure']
        )
    basetransform = BaseTransform(
        img_size=d_cfg['test_size'],
        pixel_mean=d_cfg['pixel_mean'],
        pixel_std=d_cfg['pixel_std'],
        )

    # dataset
    if args.dataset in ['ucf24', 'jhmdb21']:
        # dataset
        dataset = UCF_JHMDB_Dataset(
            data_root=d_cfg['data_root'],
            dataset=args.dataset,
            img_size=d_cfg['train_size'],
            transform=augmentation,
            is_train=is_train,
            len_clip=d_cfg['len_clip'],
            sampling_rate=d_cfg['sampling_rate']
            )
        num_classes = dataset.num_classes

        # evaluator
        evaluator = UCF_JHMDB_Evaluator(
            data_root=d_cfg['data_root'],
            dataset=args.dataset,
            model_name=args.version,
            metric='fmap',
            img_size=d_cfg['test_size'],
            len_clip=d_cfg['len_clip'],
            batch_size=d_cfg['test_batch_size'],
            conf_thresh=0.01,
            iou_thresh=0.5,
            gt_folder=d_cfg['gt_folder'],
            save_path='./evaluator/eval_results/',
            transform=basetransform,
            collate_fn=CollateFunc()            
        )

    elif args.dataset == 'ava_v2.2':
        # dataset
        dataset = AVA_Dataset(
            cfg=d_cfg,
            is_train=True,
            img_size=d_cfg['train_size'],
            transform=augmentation,
            len_clip=d_cfg['len_clip'],
            sampling_rate=d_cfg['sampling_rate']
        )
        num_classes = 80

        # evaluator
        evaluator = AVA_Evaluator(
            d_cfg=d_cfg,
            img_size=d_cfg['test_size'],
            len_clip=d_cfg['len_clip'],
            sampling_rate=d_cfg['sampling_rate'],
            batch_size=d_cfg['test_batch_size'],
            transform=basetransform,
            collate_fn=CollateFunc(),
            full_test_on_val=False,
            version='v2.2'
            )

    else:
        print('unknow dataset !! Only support ucf24 & jhmdb21 & ava_v2.2 !!')
        exit(0)

    print('==============================')
    print('Training model on:', args.dataset)
    print('The dataset size:', len(dataset))

    if not args.eval:
        # no evaluator during training stage
        evaluator = None

    return dataset, evaluator, num_classes


def build_dataloader(args, dataset, batch_size, collate_fn=None, is_train=False):
    if is_train:
        # distributed
        if args.distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            sampler = torch.utils.data.RandomSampler(dataset)

        batch_sampler_train = torch.utils.data.BatchSampler(sampler, 
                                                            batch_size, 
                                                            drop_last=True)
        # train dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset, 
            batch_sampler=batch_sampler_train,
            collate_fn=collate_fn, 
            num_workers=args.num_workers,
            pin_memory=True
            )
    else:
        # test dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset, 
            shuffle=False,
            collate_fn=collate_fn, 
            num_workers=args.num_workers,
            drop_last=False,
            pin_memory=True
            )
    
    return dataloader
    

def load_weight(model, path_to_ckpt=None):
    if path_to_ckpt is None:
        print('No trained weight ..')
        return model
        
    checkpoint = torch.load(path_to_ckpt, map_location='cpu')
    # checkpoint state dict
    checkpoint_state_dict = checkpoint.pop("model")
    # model state dict
    model_state_dict = model.state_dict()
    # check
    for k in list(checkpoint_state_dict.keys()):
        if k in model_state_dict:
            shape_model = tuple(model_state_dict[k].shape)
            shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
            if shape_model != shape_checkpoint:
                checkpoint_state_dict.pop(k)
        else:
            checkpoint_state_dict.pop(k)
            print(k)

    model.load_state_dict(checkpoint_state_dict)
    print('Finished loading model!')

    return model


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


class CollateFunc(object):
    def __call__(self, batch):
        batch_frame_id = []
        batch_key_target = []
        batch_video_clips = []

        for sample in batch:
            key_frame_id = sample[0]
            video_clip = sample[1]
            key_target = sample[2]
            
            batch_frame_id.append(key_frame_id)
            batch_video_clips.append(video_clip)
            batch_key_target.append(key_target)

        # List [B, 3, T, H, W] -> [B, 3, T, H, W]
        batch_video_clips = torch.stack(batch_video_clips)
        
        return batch_frame_id, batch_video_clips, batch_key_target


class AVA_FocalLoss(object):
    """ Focal loss for AVA"""
    def __init__(self, device, gamma, num_classes, reduction='none'):
        with open('config/ava_categories_ratio.json', 'r') as fb:
            self.class_ratio = json.load(fb)
        self.device = device
        self.gamma = gamma
        self.num_classes = num_classes
        self.reduction = reduction
        self.class_weight = torch.zeros(80).to(device)
        self._init_class_weight()


    def _init_class_weight(self):
        for i in range(1, 81):
            self.class_weight[i - 1] = 1 - self.class_ratio[str(i)]


    def __call__(self, logits, targets):
        '''
        inputs: (N, C) -- result of sigmoid
        targets: (N, C) -- one-hot variable
        '''
        # process class pred
        inputs1 = torch.clamp(torch.softmax(logits[..., :14], dim=-1), min=1e-4, max=1 - 1e-4)
        inputs2 = torch.clamp(torch.sigmoid(logits[..., 14:]), min=1e-4, max=1 - 1e-4)

        inputs = torch.cat([inputs1, inputs2], dim=-1)

        # weight matrix
        weight_matrix = self.class_weight.expand(logits.size(0), self.num_classes)
        weight_p1 = torch.exp(weight_matrix[targets == 1])
        weight_p0 = torch.exp(1 - weight_matrix[targets == 0])

        # pos & neg output
        p_1 = inputs[targets == 1]
        p_0 = inputs[targets == 0]

        # loss
        loss1 = torch.pow(1 - p_1, self.gamma) * torch.log(p_1) * weight_p1
        loss2 = torch.pow(p_0, self.gamma) * torch.log(1 - p_0) * weight_p0
        loss = -loss1.sum() - loss2.sum()

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss


class Softmax_FocalLoss(nn.Module):
    """ Focal loss for UCF24 & JHMDB21"""
    def __init__(self, num_classes, alpha=None, gamma=2.0, reduction='none'):
        super(Softmax_FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(num_classes, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.num_classes = num_classes
        self.reduction = reduction


    def forward(self, inputs, targets):
        """
            inputs: (Tensor): [N, C]
            targets: (Tensor): [N,]
        """
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids, 1.)
        
        self.alpha = self.alpha.to(inputs.device)
        alpha = self.alpha[ids.data.view(-1)]
        
        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()

        loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p 
        
        if self.reduction == 'mean':
            loss = loss.mean()

        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss
