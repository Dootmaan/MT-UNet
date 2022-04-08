# -*- coding: utf-8 -*-

from __future__ import print_function, division
import torch.nn as nn
import torch.utils.data
import torch
from resnet import ResNetV2


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


class AttU_Net(nn.Module):
    """
    Attention Unet implementation
    Paper: https://arxiv.org/abs/1804.03999
    """

    def __init__(self, img_ch=3, output_ch=1):
        super(AttU_Net, self).__init__()
        self.resnet = ResNetV2(block_units=(3, 4, 9), width_factor=1)
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        '''
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(img_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])
        '''

        self.Up5 = up_conv(filters[4], filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = conv_block(filters[3], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block(filters[2], filters[2])

        self.Up3 = up_conv(filters[2], filters[0])
        self.Att3 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv3 = conv_block(filters[0], filters[0])

        self.Up2 = up_conv(filters[0], 32)

        '''
        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv2 = conv_block(filters[1], filters[0])
        '''
        self.Conv = nn.Conv2d(32, output_ch, kernel_size=1, stride=1, padding=0)

        # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        '''
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        # print(x5.shape)
        '''

        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        x, features = self.resnet(x)  # (1024, 14, 14)
        e1 = features[2]  # (64, 112, 112)
        e2 = features[1]  # (256, 56, 56)
        e0 = features[0]  # (512, 28, 28)

        d5 = self.Up5(x)  # (512, 28, 28)
        # print(d5.shape)
        x4 = self.Att5(g=d5, x=e0)
        #d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(x4)

        d4 = self.Up4(d5)  # (256, 56, 56)
        x3 = self.Att4(g=d4, x=e2)
        #d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(x3)

        d3 = self.Up3(d4)  # (64, 112, 112)
        x2 = self.Att3(g=d3, x=e1)
        #d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(x2)
        d3 = self.Up2(d3)
        out = self.Conv(d3)

        #  out = self.active(out)

        return out

    def load_from(self, weights):
        with torch.no_grad():
            print(0)
            res_weight = weights
            self.resnet.root.conv.weight.copy_(
                np2th(res_weight["conv_root/kernel"], conv=True))
            gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
            gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
            self.resnet.root.gn.weight.copy_(gn_weight)
            self.resnet.root.gn.bias.copy_(gn_bias)

            for bname, block in self.resnet.body.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(res_weight, n_block=bname, n_unit=uname)
