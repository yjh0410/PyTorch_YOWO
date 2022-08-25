import math
import torch
import torch.nn as nn
from ...basic.conv import Conv2d


# Channel Self Attetion Module
class CSAM(nn.Module):
    """ Channel attention module """
    def __init__(self):
        super(CSAM, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)


    def forward(self, x):
        """
            inputs :
                x : input feature maps( B x C x H x W )
            returns :
                out : attention value + input feature
                attention: B x C x C
        """
        B, C, H, W = x.size()
        # query / key / value
        query = x.view(B, C, -1)
        key = x.view(B, C, -1).permute(0, 2, 1)
        value = x.view(B, C, -1)

        # attention matrix
        energy = torch.bmm(query, key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)

        # attention
        out = torch.bmm(attention, value)
        out = out.view(B, C, H, W)

        # output
        out = self.gamma * out + x

        return out


# Spatial Self Attetion Module
class SSAM(nn.Module):
    """ Spatial attention module """
    def __init__(self):
        super(SSAM, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)


    def forward(self, x):
        """
            inputs :
                x : input feature maps( B x C x H x W )
            returns :
                out : attention value + input feature
                attention: B x HW x HW
        """
        B, C, H, W = x.size()
        # query / key / value
        query = x.view(B, C, -1).permute(0, 2, 1)
        key = x.view(B, C, -1)
        value = x.view(B, C, -1).permute(0, 2, 1)

        # attention matrix
        energy = torch.bmm(query, key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)

        # attention
        out = torch.bmm(attention, value)
        out = out.permute(0, 2, 1).contiguous()
        out = out.view(B, C, H, W)

        # output
        out = self.gamma * out + x

        return out


# Spatial Cross Attetion Module
class SCAM(nn.Module):
    """ Spatial attention module """
    def __init__(self, in_dim_1, in_dim_2, out_dim=256):
        super(SCAM, self).__init__()
        self.softmax  = nn.Softmax(dim=-1)
        self.input_proj_1 = nn.Conv2d(in_dim_1, out_dim, kernel_size=1)
        self.input_proj_2 = nn.Conv2d(in_dim_2, out_dim, kernel_size=1)


    def forward(self, x1, x2):
        """
            inputs :
                x1 : (Tensor) [B, C1, H, W]
                x2 : (Tensor) [B, C2, H, W]
            returns :
                output : (Tensor) [B, C, H, W]
        """
        x1 = self.input_proj_1(x1)
        x2 = self.input_proj_2(x2)

        B, C, H, W = x1.size()
        # query / key / value
        query = x1.view(B, C, -1).permute(0, 2, 1)
        key = x2.view(B, C, -1)
        value = x2.view(B, C, -1).permute(0, 2, 1)

        # attention matrix
        energy = torch.bmm(query, key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)

        # attention
        out = torch.bmm(attention, value)
        out = out.permute(0, 2, 1).contiguous()
        out = out.view(B, C, H, W)

        return out


# Spatial Encoder
class SpatialEncoder(nn.Module):
    def __init__(self, in_dim_1=425, in_dim_2=2048, inter_dim=256, act_type='', norm_type=''):
        super().__init__()
        self.out_dim = inter_dim * 2
        self.sam_1 = nn.Sequential(
            Conv2d(in_dim_1, inter_dim, k=1, act_type=act_type, norm_type=norm_type),
            Conv2d(inter_dim, inter_dim, k=3, p=1, act_type=act_type, norm_type=norm_type),
            SSAM(),
            Conv2d(inter_dim, inter_dim, k=3, p=1, act_type=act_type, norm_type=norm_type)
        )
        self.sam_2 = nn.Sequential(
            Conv2d(in_dim_2, inter_dim, k=1, act_type=act_type, norm_type=norm_type),
            Conv2d(inter_dim, inter_dim, k=3, p=1, act_type=act_type, norm_type=norm_type),
            SSAM(),
            Conv2d(inter_dim, inter_dim, k=3, p=1, act_type=act_type, norm_type=norm_type)
        )
        self.cam_1 = nn.Sequential(
            SCAM(in_dim_1, in_dim_2, out_dim=inter_dim),
            Conv2d(inter_dim, inter_dim, k=3, p=1, act_type=act_type, norm_type=norm_type)
        )

        self.cam_2 = nn.Sequential(
            SCAM(in_dim_1, in_dim_2, out_dim=inter_dim),
            Conv2d(inter_dim, inter_dim, k=3, p=1, act_type=act_type, norm_type=norm_type)
        )

        self.output_conv_1 = Conv2d(inter_dim*2, inter_dim*2, k=1, act_type=act_type, norm_type=norm_type)
        self.output_conv_2 = Conv2d(inter_dim*2, inter_dim*2, k=1, act_type=act_type, norm_type=norm_type)


    def forward(self, x1, x2):
        """
            x: [B, CN, H, W]
        """
        x1_1 = self.sam_1(x1)
        x1_2 = self.cam_1(x1, x2)
        x2_1 = self.sam_2(x2)
        x2_2 = self.cam_2(x2, x1)

        y1 = self.output_conv_1([x1_1, x1_2])
        y2 = self.output_conv_2([x2_1, x2_2])

        return y1, y2


# Channel Encoder
class ChannelEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, act_type='', norm_type=''):
        super().__init__()
        self.fuse_convs = nn.Sequential(
            Conv2d(in_dim, out_dim, k=1, act_type=act_type, norm_type=norm_type),
            Conv2d(out_dim, out_dim, k=3, p=1, act_type=act_type, norm_type=norm_type),
            CSAM(),
            Conv2d(out_dim, out_dim, k=3, p=1, act_type=act_type, norm_type=norm_type),
            nn.Dropout(0.1, inplace=False),
            nn.Conv2d(out_dim, out_dim, kernel_size=1)
        )

    def forward(self, x):
        """
            x: [B, CN, H, W]
        """
        # [B, CN, H, W] -> [B, C, H, W]
        x = self.fuse_convs(x)

        return x
