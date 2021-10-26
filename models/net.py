from collections import OrderedDict
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils import model_zoo
import copy
import numpy as np
from torchvision import utils
from .modules import *

class model(nn.Module):
    def __init__(self, Encoder, num_features, block_channel):
        super(model, self).__init__()
        self.E = Encoder
        self.MFF = MFF(block_channel)
        self.R = R(block_channel)


    def forward(self, x):
        x_block1, x_block2, x_block3, x_block4 = self.E(x)
        x_mff = self.MFF(x_block1, x_block2, x_block3, x_block4,[x_block1.size(2)*2,x_block1.size(3)*2])
        out = self.R(x_mff)
        out = F.upsample(out, size=[228,304], mode='bilinear', align_corners=True)

        return out

