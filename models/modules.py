from collections import OrderedDict
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils import model_zoo
import copy
import numpy as np


class _UpProjection(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super(_UpProjection, self).__init__()

        self.conv1 = nn.Conv2d(num_input_features, num_output_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_output_features)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, size):

        x = self.relu(self.bn1(self.conv1(x)))
        x = F.upsample(x, size=size, mode='bilinear', align_corners=True)

        return x


class E_mvnet2(nn.Module):

    def __init__(self, original_model):
        super(E_mvnet2, self).__init__()
        self.mv2 = original_model.features[:17]

    def forward(self, x):
        x_block1 =self.mv2[0:4](x)
        x_block2 = self.mv2[4:7](x_block1)
        x_block3 = self.mv2[7:14](x_block2)
        x_block4 = self.mv2[14:](x_block3)

        return x_block1, x_block2, x_block3, x_block4


class E_resnet(nn.Module):

    def __init__(self, original_model, num_features=2048):
        super(E_resnet, self).__init__()
        self.conv1 = original_model.conv1
        self.bn1 = original_model.bn1
        self.relu = original_model.relu
        self.maxpool = original_model.maxpool

        self.layer1 = original_model.layer1
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x_block1 = self.layer1(x)
        x_block2 = self.layer2(x_block1)
        x_block3 = self.layer3(x_block2)
        x_block4 = self.layer4(x_block3)

        return x_block1, x_block2, x_block3, x_block4


class MFF(nn.Module):

    def __init__(self, block_channel, num_features=64, num_output_features=16):
        super(MFF, self).__init__()

        self.att1 = self.attention(block_channel[0], block_channel[0] // 16)
        self.att2 = self.attention(block_channel[1], block_channel[1] // 16)
        self.att3 = self.attention(block_channel[2], block_channel[2] // 16)
        self.att4 = self.attention(block_channel[3], block_channel[3] // 16)

        self.up1 = _UpProjection(block_channel[0], num_output_features)
        self.up2 = _UpProjection(block_channel[1], num_output_features)
        self.up3 = _UpProjection(block_channel[2], num_output_features)
        self.up4 = _UpProjection(block_channel[3], num_output_features)

        self.conv = nn.Conv2d(num_features, num_features,
                              kernel_size=5, stride=1, padding=2, bias=False)
        self.bn = nn.BatchNorm2d(num_features)

    def attention(self, features1, features2):
        prior = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        conv1 = nn.Conv2d(features1, features2, kernel_size=1, bias=False)
        # bn = nn.BatchNorm2d(features)
        relu = nn.ReLU()
        conv2 = nn.Conv2d(features2, features1, kernel_size=1, bias=False)
        sigmoid = nn.Sigmoid()
        return nn.Sequential(prior, conv1, relu, conv2, sigmoid)

    def forward(self, x_block1, x_block2, x_block3, x_block4, size):
        x_att1 = self.att1(x_block1)
        x_att2 = self.att2(x_block2)
        x_att3 = self.att3(x_block3)
        x_att4 = self.att4(x_block4)

        x_m1 = self.up1(x_block1 * x_att1, size)
        x_m2 = self.up2(x_block2 * x_att2, size)
        x_m3 = self.up3(x_block3 * x_att3, size)
        x_m4 = self.up4(x_block4 * x_att4, size)

        x = self.bn(self.conv(torch.cat((x_m1, x_m2, x_m3, x_m4), 1)))
        x = F.relu(x)

        return x



class R(nn.Module):

    def __init__(self, block_channel, num_features = 64):

        super(R, self).__init__()
        self.conv0 = nn.Conv2d(num_features, num_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn0 = nn.BatchNorm2d(num_features)

        self.conv2 = nn.Conv2d(
            num_features, 1, kernel_size=5, stride=1, padding=2, bias=True)

    def forward(self, x):
        x0 = self.conv0(x)
        x0 = self.bn0(x0)
        x0 = F.relu(x0)

        out = self.conv2(x0)

        return out

