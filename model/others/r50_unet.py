#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: zerone_xie
# datetime:2021/9/18 23:12
# software: PyCharm
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


class resunet(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_ch=3, out_ch=1):
        super(resunet, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]  # (64, 128, 256, 512, 1024)
        '''
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])
        '''
        self.encoder = ResNetV2(block_units=(3, 4, 9), width_factor=1)
        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[3], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[2], filters[2])

        #self.Up3 = up_conv(filters[2], filters[1])
        #self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[2], filters[0])
        self.Up_conv2 = conv_block(filters[0], filters[0])

        self.Up1 = up_conv(filters[0], 32)
        self.Conv = nn.Conv2d(32, out_ch, kernel_size=1, stride=1, padding=0)

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
        '''
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        e3, features = self.encoder(x)  # (1024, 14, 14)
        e2 = features[2]  # (64, 112, 112)
        e1 = features[1]  # (256, 56, 56)
        e0 = features[0]  # (512, 28, 28)

        d5 = self.Up5(e3)  # (512, 28, 28)
        #d5 = torch.cat((e0, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)  # (256, 56, 56)
        #d4 = torch.cat((e1, d4), dim=1)
        d4 = self.Up_conv4(d4)
        '''
        d3 = self.Up3(d4)  
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        '''
        d2 = self.Up2(d4)
        #d2 = torch.cat((e2, d2), dim=1)
        d2 = self.Up_conv2(d2)   # (16, 64, 112, 112)
        d2 = self.Up1(d2)
        out = self.Conv(d2)

        # d1 = self.active(out)

        return out

    def load_from(self, weights):
        with torch.no_grad():
            print(0)
            res_weight = weights
            self.encoder.root.conv.weight.copy_(
                np2th(res_weight["conv_root/kernel"], conv=True))
            gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
            gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
            self.encoder.root.gn.weight.copy_(gn_weight)
            self.encoder.root.gn.bias.copy_(gn_bias)

            for bname, block in self.encoder.body.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(res_weight, n_block=bname, n_unit=uname)