#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: zerone_xie
# datetime:2021/8/7 13:31
# software: PyCharm
import math

import numpy as np
import torch
from torch import nn


class ConvBNReLU(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride=1, padding=1, activation=True):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU()
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation:
            x = self.relu(x)
        return x


class DoubleConv(nn.Module):

    def __init__(self, cin, cout):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            ConvBNReLU(cin, cout, 3, 1, padding=1),
            ConvBNReLU(cout, cout, 3, stride=1, padding=1, activation=False)
        )
        self.conv1 = nn.Conv2d(cout, cout, 1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = self.conv(x)
        h = x
        x = self.conv1(x)
        x = self.bn(x)
        x = h + x
        x = self.relu(x)
        return x


class U_encoder(nn.Module):

    def __init__(self):
        super(U_encoder, self).__init__()
        self.res1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.res2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.res3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)


    def forward(self, x):
        features = []
        x = self.res1(x)
        features.append(x)  # (224, 224, 64)
        x = self.pool1(x)

        x = self.res2(x)
        features.append(x)  # (112, 112, 128)
        x = self.pool2(x)

        x = self.res3(x)
        features.append(x)  # (56, 56, 256)
        x = self.pool3(x)

        return x, features


class U_decoder(nn.Module):

    def __init__(self):
        super(U_decoder, self).__init__()
        self.trans1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.res1 = DoubleConv(512, 256)
        self.trans2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.res2 = DoubleConv(256, 128)
        self.trans3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.res3 = DoubleConv(128, 64)

    def forward(self, x, feature):

        x = self.trans1(x)  # (56, 56, 256)
        x = torch.cat((feature[2], x), dim=1)
        x = self.res1(x)  # (56, 56, 256)
        x = self.trans2(x)  # (112, 112, 128)
        x = torch.cat((feature[1], x), dim=1)
        x = self.res2(x)  # (112, 112, 128)
        x = self.trans3(x)  # (224, 224, 64)
        x = torch.cat((feature[0], x), dim=1)
        x = self.res3(x)
        return x


class MEAttention(nn.Module):
    '''
    首先经过linear层计算query,然后分别计算k,v计算attention
    '''
    def __init__(self, dim, configs):
        super(MEAttention, self).__init__()
        self.num_heads = configs["head"]
        self.coef = 4
        self.query_liner = nn.Linear(dim, dim * self.coef)
        self.num_heads = self.coef * self.num_heads
        self.k = 256 // self.coef
        self.linear_0 = nn.Linear(dim * self.coef // self.num_heads, self.k)
        self.linear_1 = nn.Linear(self.k, dim * self.coef // self.num_heads)

        self.proj = nn.Linear(dim * self.coef, dim)

    def forward(self, x):
        B, N, C = x.shape
        x = self.query_liner(x)
        x = x.view(B, N, self.num_heads, -1).permute(0, 2, 1, 3)  #（1， 32， 225， 32）

        attn = self.linear_0(x)

        attn = attn.softmax(dim=-2)
        attn = attn / (1e-9 + attn.sum(dim=-1, keepdim=True))

        x = self.linear_1(attn).permute(0, 2, 1, 3).reshape(B, N, -1)

        x = self.proj(x)

        return x

class Attention(nn.Module):
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.num_attention_heads = 8
        self.attention_head_size = int(dim / 8)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(dim, self.all_head_size)
        self.key = nn.Linear(dim, self.all_head_size)
        self.value = nn.Linear(dim, self.all_head_size)

        self.out = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(0.0)
        self.proj_dropout = nn.Dropout(0.0)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        #weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output

class MLP(nn.Module):

    def __init__(self, dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(dim, dim*4)
        self.fc2 = nn.Linear(dim*4, dim)
        self.act = nn.functional.gelu
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class EAmodule(nn.Module):

    def __init__(self, dim):
        super(EAmodule, self).__init__()
        self.SlayerNorm = nn.LayerNorm(dim, eps=1e-6)
        self.ElayerNorm = nn.LayerNorm(dim, eps=1e-6)
        #self.CSAttention = CSAttention(dim, configs)
        #self.EAttention = MEAttention(dim, configs)
        self.MSA = Attention(dim)
        self.MLP = MLP(dim)
    def forward(self, x):
        h = x  # (B, N, H)
        x = self.SlayerNorm(x)

        x = self.MSA(x)  # padding 到right_size
        x = h + x

        h = x
        x = self.ElayerNorm(x)

        #x = self.EAttention(x)
        x = self.MLP(x)
        x = h + x

        return x


class DecoderStem(nn.Module):

    def __init__(self):
        super(DecoderStem, self).__init__()
        self.block = U_decoder()

    def forward(self, x, features):
        x = self.block(x, features)
        return x


class Stem(nn.Module):

    def __init__(self):
        super(Stem, self).__init__()
        self.model = U_encoder()
        self.trans_dim = ConvBNReLU(256, 256, 1, 1, 0) #out_dim, model_dim
        self.position_embedding = nn.Parameter(torch.zeros((1, 784, 256)))  #缺少n_patches, hiddensize参数

    def forward(self, x):

        x, features = self.model(x)  # (1, 512, 28, 28)
        x = self.trans_dim(x)  # (B, C, H, W) (1, 256, 28, 28)
        x = x.flatten(2)    # (B, H, N)  (1, 256, 28*28)
        x = x.transpose(-2, -1)  #  (B, N, H)
        x = x + self.position_embedding
        return x, features      #(B, N, H)


class encoder_block(nn.Module):

    def __init__(self, dim):
        super(encoder_block, self).__init__()
        self.block = nn.ModuleList([
            EAmodule(dim),
            EAmodule(dim),
            ConvBNReLU(dim, dim * 2, 2, stride=2, padding=0)
        ])

    def forward(self, x):
        x = self.block[0](x)
        x = self.block[1](x)
        B, N, C = x.shape
        h, w = int(np.sqrt(N)), int(np.sqrt(N))
        x = x.view(B, h, w, C).permute(0, 3, 1, 2)  # (1, 256, 28, 28) B, C, H, W
        skip = x
        x = self.block[2](x)   # (14, 14, 256)
        return x, skip


class decoder_block(nn.Module):

    def __init__(self, dim, flag):
        super(decoder_block, self).__init__()
        self.flag = flag
        if not self.flag:
            self.block = nn.ModuleList([
                nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2, padding=0),
                nn.Conv2d(dim, dim // 2, kernel_size=1, stride=1),
                EAmodule(dim // 2),
                EAmodule(dim // 2),
            ])
        else:
            self.block = nn.ModuleList([
                nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2, padding=0),
                EAmodule(dim),
                EAmodule(dim)
            ])

    def forward(self, x, skip):
        if not self.flag:
            x = self.block[0](x)
            x = torch.cat((x, skip), dim=1)
            x = self.block[1](x)
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1).contiguous().view(B, -1, C)
            x = self.block[2](x)
            x = self.block[3](x)
        else:
            x = self.block[0](x)
            x = torch.cat((x, skip), dim=1)
            x = x.permute(0, 2, 3, 1)
            B, H, W, C = x.shape
            x = x.view(B, -1, C)
            x = self.block[1](x)
            x = self.block[2](x)
        return x


class EAUnet(nn.Module):

    def __init__(self):
        super(EAUnet, self).__init__()
        self.stem = Stem()
        self.encoder = nn.ModuleList()
        self.bottleneck = nn.Sequential(
            EAmodule(configs["bottleneck"]),
            EAmodule(configs["bottleneck"])
        )
        self.decoder = nn.ModuleList()

        self.decoder_stem = DecoderStem()
        for i in range(len(configs["encoder"])):
            dim = configs["encoder"][i]
            self.encoder.append(encoder_block(dim))
        for i in range(len(configs["decoder"])-1):
            dim = configs["decoder"][i]
            self.decoder.append(decoder_block(dim, False))
        self.decoder.append(decoder_block(configs["decoder"][-1], True))
        self.SegmentationHead = nn.Conv2d(64, 4, 1)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x, features = self.stem(x)  #(B, N, C) (1, 196, 256)
        skips = []
        for i in range(len(self.encoder)):
            x, skip = self.encoder[i](x)
            skips.append(skip)
            B, C, H, W = x.shape    #  (1, 512, 8, 8)
            x = x.permute(0, 2, 3, 1).contiguous().view(B, -1, C)  # （B, N, C)
        x = self.bottleneck(x)   # (1, 25, 1024)
        B, N, C = x.shape
        x = x.view(B, int(np.sqrt(N)), -1, C).permute(0, 3, 1, 2)
        for i in range(len(self.decoder)):
            x = self.decoder[i](x, skips[len(self.decoder) - i - 1])  # (B， N, C）
            B, N, C = x.shape
            x = x.view(B, int(np.sqrt(N)), int(np.sqrt(N)), C).permute(0, 3, 1, 2)

        x = self.decoder_stem(x, features)
        x = self.SegmentationHead(x)
        return x


configs = {
    "win_size": 4,
    "head": 8,
    "axis": [28, 16, 8],
    "encoder": [256, 512],
    "bottleneck": 1024,
    "decoder": [1024, 512],
    "decoder_stem": [(256, 512), (256, 256), (128, 64), 32]
}