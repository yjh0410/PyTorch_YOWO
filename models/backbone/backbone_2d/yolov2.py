import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url


__all__ = ['YOLOv2', 'build_yolov2']


model_urls = {
    'yolov2': 'https://github.com/yjh0410/PyTorch_YOWO/releases/download/yowo-weight/yolov2.pth'
}


# Basic Module
class Conv_BN_LeakyReLU(nn.Module):
    def __init__(self, in_dim, out_dim, ksize, padding=0, stride=1, dilation=1):
        super(Conv_BN_LeakyReLU, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, ksize, padding=padding, stride=stride, dilation=dilation),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.convs(x)


class ReOrg(nn.Module):
    def __init__(self, stride):
        super(ReOrg, self).__init__()
        self.stride = stride

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        _height, _width = height // self.stride, width // self.stride
        
        x = x.view(batch_size, channels, _height, self.stride, _width, self.stride).transpose(3, 4).contiguous()
        x = x.view(batch_size, channels, _height * _width, self.stride * self.stride).transpose(2, 3).contiguous()
        x = x.view(batch_size, channels, self.stride * self.stride, _height, _width).transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, _height, _width)

        return x


# Backbone of YOLOv2
class DarkNet19(nn.Module):
    def __init__(self):        
        super(DarkNet19, self).__init__()
        # backbone network : DarkNet-19
        # output : stride = 2, c = 32
        self.conv_1 = nn.Sequential(
            Conv_BN_LeakyReLU(3, 32, 3, 1),
            nn.MaxPool2d((2,2), 2),
        )

        # output : stride = 4, c = 64
        self.conv_2 = nn.Sequential(
            Conv_BN_LeakyReLU(32, 64, 3, 1),
            nn.MaxPool2d((2,2), 2)
        )

        # output : stride = 8, c = 128
        self.conv_3 = nn.Sequential(
            Conv_BN_LeakyReLU(64, 128, 3, 1),
            Conv_BN_LeakyReLU(128, 64, 1),
            Conv_BN_LeakyReLU(64, 128, 3, 1),
            nn.MaxPool2d((2,2), 2)
        )

        # output : stride = 8, c = 256
        self.conv_4 = nn.Sequential(
            Conv_BN_LeakyReLU(128, 256, 3, 1),
            Conv_BN_LeakyReLU(256, 128, 1),
            Conv_BN_LeakyReLU(128, 256, 3, 1),
        )

        # output : stride = 16, c = 512
        self.maxpool_4 = nn.MaxPool2d((2, 2), 2)
        self.conv_5 = nn.Sequential(
            Conv_BN_LeakyReLU(256, 512, 3, 1),
            Conv_BN_LeakyReLU(512, 256, 1),
            Conv_BN_LeakyReLU(256, 512, 3, 1),
            Conv_BN_LeakyReLU(512, 256, 1),
            Conv_BN_LeakyReLU(256, 512, 3, 1),
        )
        
        # output : stride = 32, c = 1024
        self.maxpool_5 = nn.MaxPool2d((2, 2), 2)
        self.conv_6 = nn.Sequential(
            Conv_BN_LeakyReLU(512, 1024, 3, 1),
            Conv_BN_LeakyReLU(1024, 512, 1),
            Conv_BN_LeakyReLU(512, 1024, 3, 1),
            Conv_BN_LeakyReLU(1024, 512, 1),
            Conv_BN_LeakyReLU(512, 1024, 3, 1)
        )


    def forward(self, x):
        c1 = self.conv_1(x)
        c2 = self.conv_2(c1)
        c3 = self.conv_4(self.conv_3(c2))
        c4 = self.conv_5(self.maxpool_4(c3))
        c5 = self.conv_6(self.maxpool_5(c4))

        outputs = {
            'c3': c3,
            'c4': c4,
            'c5': c5
        }

        return outputs


# YOLOv2
class YOLOv2(nn.Module):
    def __init__(self):
        super(YOLOv2, self).__init__()

        # backbone
        self.backbone = DarkNet19()
        
        # neck
        self.convsets_1 = nn.Sequential(
            Conv_BN_LeakyReLU(1024, 1024, ksize=3, padding=1),
            Conv_BN_LeakyReLU(1024, 1024, ksize=3, padding=1)
        )

        # reorg
        self.route_layer = Conv_BN_LeakyReLU(512, 64, ksize=1)
        self.reorg = ReOrg(stride=2)

        # output
        self.convsets_2 = Conv_BN_LeakyReLU(1280, 1024, ksize=3, padding=1)
        self.pred = nn.Conv2d(1024, 425, kernel_size=1) # 425 = 5x(80 + 5)
        

    def forward(self, x):
        """
        Input:
            x: (Tensor) [B, C_in, H_in, W_in]
        Output:
            y: (Tensor) [B, C_out, H_out, W_out]
        """
        # backbone
        outputs = self.backbone(x)
        c4, c5 = outputs['c4'], outputs['c5']
        p5 = self.convsets_1(c5)

        # reorg
        p4 = self.reorg(self.route_layer(c4))
        p5 = torch.cat([p4, p5], dim=1)

        # output
        y = self.convsets_2(p5)
        y = self.pred(y)

        return y


# build YOLOv2
def build_yolov2(pretrained):
    model = YOLOv2()
    bk_dim = 425

    # Load COCO pretrained weight
    if pretrained:
        print('Loading pretrained weight ...')
        checkpoint_state_dict = load_state_dict_from_url(
            model_urls['yolov2'],
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
