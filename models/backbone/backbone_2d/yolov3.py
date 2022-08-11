import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url


__all__ = ['YOLOv3', 'build_yolov3']


model_urls = {
    'yolov3': 'https://github.com/yjh0410/PyTorch_YOWO/releases/download/yowo-weight/yolov3.pth'
}


# Basic Module
class Conv_BN_LeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, padding=0, stride=1, dilation=1):
        super(Conv_BN_LeakyReLU, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, padding=padding, stride=stride, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.convs(x)


class ConvBlocks(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        inter_dim = out_dim *2
        self.convs = nn.Sequential(
            Conv_BN_LeakyReLU(in_dim, out_dim, ksize=1),
            Conv_BN_LeakyReLU(out_dim, inter_dim, ksize=3, padding=1),
            Conv_BN_LeakyReLU(inter_dim, out_dim, ksize=1),
            Conv_BN_LeakyReLU(out_dim, inter_dim, ksize=3, padding=1),
            Conv_BN_LeakyReLU(inter_dim, out_dim, ksize=1)
        )

    def forward(self, x):
        return self.convs(x)


class UpSample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corner=None):
        super(UpSample, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corner = align_corner

    def forward(self, x):
        return torch.nn.functional.interpolate(input=x, 
                                               size=self.size, 
                                               scale_factor=self.scale_factor, 
                                               mode=self.mode, 
                                               align_corners=self.align_corner
                                               )


# Residual Block
class ResBlock(nn.Module):
    def __init__(self, ch, nblocks=1):
        super().__init__()
        self.module_list = nn.ModuleList()
        for _ in range(nblocks):
            resblock_one = nn.Sequential(
                Conv_BN_LeakyReLU(ch, ch//2, 1),
                Conv_BN_LeakyReLU(ch//2, ch, 3, padding=1)
            )
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            x = module(x) + x
        return x


# DarkNet-53
class DarkNet_53(nn.Module):
    """
    DarkNet-53.
    """
    def __init__(self):
        super(DarkNet_53, self).__init__()
        # stride = 2
        self.layer_1 = nn.Sequential(
            Conv_BN_LeakyReLU(3, 32, 3, padding=1),
            Conv_BN_LeakyReLU(32, 64, 3, padding=1, stride=2),
            ResBlock(64, nblocks=1)
        )
        # stride = 4
        self.layer_2 = nn.Sequential(
            Conv_BN_LeakyReLU(64, 128, 3, padding=1, stride=2),
            ResBlock(128, nblocks=2)
        )
        # stride = 8
        self.layer_3 = nn.Sequential(
            Conv_BN_LeakyReLU(128, 256, 3, padding=1, stride=2),
            ResBlock(256, nblocks=8)
        )
        # stride = 16
        self.layer_4 = nn.Sequential(
            Conv_BN_LeakyReLU(256, 512, 3, padding=1, stride=2),
            ResBlock(512, nblocks=8)
        )
        # stride = 32
        self.layer_5 = nn.Sequential(
            Conv_BN_LeakyReLU(512, 1024, 3, padding=1, stride=2),
            ResBlock(1024, nblocks=4)
        )


    def forward(self, x, targets=None):
        c1 = self.layer_1(x)
        c2 = self.layer_2(c1)
        c3 = self.layer_3(c2)
        c4 = self.layer_4(c3)
        c5 = self.layer_5(c4)

        return c3, c4, c5


# YOLOv3
class YOLOv3(nn.Module):
    def __init__(self):

        super(YOLOv3, self).__init__()
        self.bk_dims = [256, 512, 1024]
        c3, c4, c5 = self.bk_dims

        # backbone
        self.backbone = DarkNet_53()
        
        # head
        # P3/8-small
        self.head_convblock_0 = ConvBlocks(in_dim=c5, out_dim=c5//2)
        self.head_conv_0 = Conv_BN_LeakyReLU(c5//2, c4//2, ksize=1)
        self.head_upsample_0 = UpSample(scale_factor=2)
        self.head_conv_1 = Conv_BN_LeakyReLU(c5//2, c5, ksize=3, padding=1)

        # P4/16-medium
        self.head_convblock_1 = ConvBlocks(in_dim=c4 + c4//2, out_dim=c4//2)
        self.head_conv_2 = Conv_BN_LeakyReLU(c4//2, c3//2, ksize=1)
        self.head_upsample_1 = UpSample(scale_factor=2)
        self.head_conv_3 = Conv_BN_LeakyReLU(c4//2, c4, ksize=3, padding=1)

        # P8/32-large
        self.head_convblock_2 = ConvBlocks(in_dim=c3 + c3//2, out_dim=c3//2)
        self.head_conv_4 = Conv_BN_LeakyReLU(c3//2, c3, ksize=3, padding=1)

        # det conv
        self.head_det_1 = nn.Conv2d(c3, 255, kernel_size=1)
        self.head_det_2 = nn.Conv2d(c4, 255, kernel_size=1)
        self.head_det_3 = nn.Conv2d(c5, 255, kernel_size=1)


    def forward(self, x):
        # backbone
        c3, c4, c5 = self.backbone(x)

        # head
        # p5/32
        p5 = self.head_convblock_0(c5)
        p5_up = self.head_upsample_0(self.head_conv_0(p5))
        p5 = self.head_conv_1(p5)

        # p4/16
        p4 = self.head_convblock_1(torch.cat([c4, p5_up], dim=1))
        p4_up = self.head_upsample_1(self.head_conv_2(p4))
        p4 = self.head_conv_3(p4)

        # P3/8
        p3 = self.head_convblock_2(torch.cat([c3, p4_up], dim=1))
        p3 = self.head_conv_4(p3)

        # det
        y1 = self.head_det_1(p3)
        y2 = self.head_det_2(p4)
        y3 = self.head_det_3(p5)

        outputs = [y1, y2, y3]
        
        return outputs


# build YOLOv3
def build_yolov3(pretrained):
    model = YOLOv3()
    bk_dim = [255, 255, 255]

    # Load COCO pretrained weight
    if pretrained:
        print('Loading pretrained weight ...')
        checkpoint_state_dict = load_state_dict_from_url(
            model_urls['yolov3'],
            map_location='cpu'
            )
        # model state dict
        model_state_dict = model.state_dict()
        # check
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    print(k)
                    checkpoint_state_dict.pop(k)
            else:
                checkpoint_state_dict.pop(k)
                print(k)

        model.load_state_dict(checkpoint_state_dict, strict=False)

    return model, bk_dim
