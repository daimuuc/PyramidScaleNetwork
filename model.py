# -*- coding: <encoding name> -*-
"""
PSNet model
"""
from __future__ import print_function, division
import torch.nn as nn
import torch
from torch.utils import model_zoo
import numpy as np
import torch.nn.functional as F


################################################################################
# PSNet
################################################################################
class PSNet(nn.Module):
    def __init__(self):
        super(PSNet, self).__init__()
        self.vgg = VGG()
        self.dmp = BackEnd()

        self._load_vgg()

    def forward(self, input):
        input = self.vgg(input)
        dmp_out = self.dmp(*input)

        return dmp_out

    def _load_vgg(self):
        state_dict = model_zoo.load_url('https://download.pytorch.org/models/vgg16_bn-6c64b313.pth')
        old_name = [0, 1, 3, 4, 7, 8, 10, 11, 14, 15, 17, 18, 20, 21, 24, 25, 27, 28, 30, 31, 34, 35, 37, 38, 40, 41]
        new_name = ['1_1', '1_2', '2_1', '2_2', '3_1', '3_2', '3_3', '4_1', '4_2', '4_3', '5_1', '5_2', '5_3']
        new_dict = {}
        for i in range(10):
            new_dict['conv' + new_name[i] + '.conv.weight'] = \
                state_dict['features.' + str(old_name[2 * i]) + '.weight']
            new_dict['conv' + new_name[i] + '.conv.bias'] = \
                state_dict['features.' + str(old_name[2 * i]) + '.bias']
            new_dict['conv' + new_name[i] + '.bn.weight'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.weight']
            new_dict['conv' + new_name[i] + '.bn.bias'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.bias']
            new_dict['conv' + new_name[i] + '.bn.running_mean'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.running_mean']
            new_dict['conv' + new_name[i] + '.bn.running_var'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.running_var']

        self.vgg.load_state_dict(new_dict)


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1_1 = BaseConv(3, 64, 3, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv1_2 = BaseConv(64, 64, 3, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_1 = BaseConv(64, 128, 3, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_2 = BaseConv(128, 128, 3, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_1 = BaseConv(128, 256, 3, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_2 = BaseConv(256, 256, 3, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_3 = BaseConv(256, 256, 3, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_1 = BaseConv(256, 512, 3, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_2 = BaseConv(512, 512, 3, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_3 = BaseConv(512, 512, 3, 1, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, input):
        input = self.conv1_1(input)
        conv1_2 = self.conv1_2(input)

        input = self.pool(conv1_2)
        input = self.conv2_1(input)
        conv2_2 = self.conv2_2(input)

        input = self.pool(conv2_2)
        input = self.conv3_1(input)
        input = self.conv3_2(input)
        conv3_3 = self.conv3_3(input)

        input = self.pool(conv3_3)
        input = self.conv4_1(input)
        input = self.conv4_2(input)
        conv4_3 = self.conv4_3(input)

        return conv1_2, conv2_2, conv3_3, conv4_3


class BackEnd(nn.Module):
    def __init__(self):
        super(BackEnd, self).__init__()

        self.dense1 = DenseModule(512)
        self.dense2 = DenseModule(512)
        self.dense3 = DenseModule(512)

        self.conv1 = BaseConv(512, 256, 3, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2 = BaseConv(256, 128, 3, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3 = BaseConv(128, 64, 3, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4 = BaseConv(64, 1, 1, 1, activation=None, use_bn=False)

    def forward(self, *input):
        conv1_2, conv2_2, conv3_3, conv4_3 = input

        input, attention_map_1 = self.dense1(conv4_3)
        input, attention_map_2 = self.dense2(input)
        input, attention_map_3 = self.dense3(input)

        input = self.conv1(input)
        input = self.conv2(input)
        input = self.conv3(input)
        input = self.conv4(input)

        return input, attention_map_1, attention_map_2, attention_map_3


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.conv1 = BaseConv(in_planes, round(in_planes // ratio), 1, 1, activation=nn.ReLU(), use_bn=False)
        self.conv2 = BaseConv(round(in_planes // ratio), in_planes, 1, 1, activation=nn.Sigmoid(), use_bn=False)

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        return out


class DenseModule(nn.Module):
    def __init__(self, in_channels):
        super(DenseModule, self).__init__()

        self.conv3x3 = nn.Sequential(
            BaseConv(in_channels, in_channels // 4, 1, 1, activation=nn.ReLU(), use_bn=True),
            BaseConv(in_channels // 4, in_channels // 4, 3, 1, 1, activation=nn.ReLU(), use_bn=True))
        self.conv5x5 = nn.Sequential(
            BaseConv(in_channels, in_channels // 4, 1, 1, activation=nn.ReLU(), use_bn=True),
            BaseConv(in_channels // 4, in_channels // 4, 3, 1, 2, 2, activation=nn.ReLU(), use_bn=True))
        self.conv7x7 = nn.Sequential(
            BaseConv(in_channels, in_channels // 4, 1, 1, activation=nn.ReLU(), use_bn=True),
            BaseConv(in_channels // 4, in_channels // 4, 3, 1, 3, 3, activation=nn.ReLU(), use_bn=True))
        self.conv9x9 = nn.Sequential(
            BaseConv(in_channels, in_channels // 4, 1, 1, activation=nn.ReLU(), use_bn=True),
            BaseConv(in_channels // 4, in_channels // 4, 3, 1, 4, 4, activation=nn.ReLU(), use_bn=True))

        self.conv1 = BaseConv(in_channels // 2, in_channels // 4, 3, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2 = BaseConv(in_channels // 2, in_channels // 4, 3, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3 = BaseConv(in_channels // 2, in_channels // 4, 3, 1, 1, activation=nn.ReLU(), use_bn=True)

        self.att = ChannelAttention(in_channels)

        self.conv = BaseConv(in_channels, in_channels, 3, 1, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, input):
        conv3x3 = self.conv3x3(input)
        conv5x5 = self.conv5x5(input)
        conv7x7 = self.conv7x7(input)
        conv9x9 = self.conv9x9(input)

        conv5x5 = self.conv1(torch.cat((conv3x3, conv5x5), dim=1))
        conv7x7 = self.conv2(torch.cat((conv5x5, conv7x7), dim=1))
        conv9x9 = self.conv3(torch.cat((conv7x7, conv9x9), dim=1))

        att = self.att(input)

        out = self.conv(torch.cat((conv3x3, conv5x5, conv7x7, conv9x9), dim=1))

        attention_map = torch.cat((torch.mean(conv3x3, dim=1, keepdim=True),
                                   torch.mean(conv5x5, dim=1, keepdim=True),
                                   torch.mean(conv7x7, dim=1, keepdim=True),
                                   torch.mean(conv9x9, dim=1, keepdim=True)), dim=1)

        return out * att, attention_map


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, padding=0, dilation=1, activation=None,
                 use_bn=False):
        super(BaseConv, self).__init__()
        self.use_bn = use_bn
        self.activation = activation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, padding, dilation)
        self.conv.weight.data.normal_(0, 0.01)
        self.conv.bias.data.zero_()
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, input):
        input = self.conv(input)
        if self.use_bn:
            input = self.bn(input)
        if self.activation:
            input = self.activation(input)

        return input