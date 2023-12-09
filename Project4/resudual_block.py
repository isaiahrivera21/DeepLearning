from typing import Any
import tensorflow as tf
from conv2d import Conv2d
from group_norm import GroupNorm


class ResidualBlock(tf.Module):
    def __init__(self, filter, stride, C, G):
        self.filter = filter
        self.stride = stride
        self.C = C
        self.G = G
        self.conv = Conv2d(filter, stride)
        self.groupnorm = GroupNorm(C, G)

    def __call__(self, x):
        x_r = x
        x = self.conv(x)
        x = self.groupnorm(x)
        x = tf.nn.relu(x)
        x = self.conv(x)
        x = self.groupnorm(x)
        x_l = x_r + x
        return x_l
