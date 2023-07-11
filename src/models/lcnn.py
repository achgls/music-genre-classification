import torch
import torch.nn as nn

from typing import Union, Tuple


class MFMConv2D(nn.Module):
    """
    Combines a conv2D layer with MFM 2/1 activation in a single module
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int]] = 3,
            stride: Union[int, Tuple[int]] = 1,
            padding: Union[str, int, Tuple[int]] = "same",
    ):
        super().__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels*2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

    def forward(self, x):
        x = self.conv(x)
        # (N, 2C, H, W)
        # Split in half along the channel axis
        x1, x2 = torch.split(x, self.out_channels, dim=1)
        # gives two tensors (N, C, H, W)
        # computes element-wise maximum of the 4-d tensor
        return torch.maximum(x1, x2)


class MaxFeatureMap21(nn.Module):
    """
    General case of the MFM 2/1 activation (works for both MLP and CNN)
    :param units: number of input units / channels
    :return mfm21: a tensor with half as many channels / units as input
    """
    def __init__(self, units):
        super().__init__()
        assert units % 2 == 0
        self.units = units

    def forward(self, x):
        x1, x2 = torch.split(x, self.units // 2, dim=1)
        return torch.maximum(x1, x2)


class MaxFeatureMap32(nn.Module):
    """
    General case of the MFM 3/2 activation (works for both MLP and CNN)
    :param units: number of input units / channels
    :return mfm32: a tensor with the 2/3 of the input units / channels
    """
    def __init__(self, units):
        super().__init__()
        assert units % 3 == 0
        self.units = units

    def forward(self, x):
        x1, x2, x3 = torch.split(x, self.units // 3, dim=1)
        x = torch.stack((x1, x2, x3), dim=1)
        x = torch.topk(x, 2, dim=1)[0]
        x = torch.cat(torch.split(x, 1, dim=1), dim=2).squeeze(1)
        return x


class TopKFeatureMap(nn.Module):
    """
    Most general case of the Max Feature Map (MFM), returning the K filters corresponding to the top K values
    contained in the N inpout filters.
    1st output filter contains the largest elements of the N input filters.
    2nd output filter contains the 2nd largest elements of the N input filters, and so on...
    """
    def __init__(self, n, k):
        super().__init__()
        assert k % n == 0, "Input # of units / channels must be a multiple of k"
        self.k = k

    def forward(self, x):
        xx = torch.chunk(x, self.k, dim=1)
        x = torch.stack(xx, dim=1)
        xx = torch.topk(x, self.k, dim=1)[0]
        x = torch.cat(torch.split(xx, 1, dim=1), dim=2).squeeze(1)
        return x


class LCNN(nn.Module):
    def __init__(self):
        super().__init__()

    def _make_layer(self):
        ...

    def forward(self, x):
        ...


class MFMResBlock(nn.Module):
    def __init__(self, layers, in_channels, out_channels, kernel_size=3, stride=1, padding='same'):
        super().__init__()
        self.layers = layers
        self.res_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding)
        self.conv_expand = nn.Conv2d(in_channels, in_channels * 2, kernel_size=1)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels * 2, kernel_size)
        self.mfm1 = MaxFeatureMap21(in_channels * 2)
        self.mfm2 = MaxFeatureMap21(out_channels * 2)

    def forward(self, x):
        identity = x

        for _ in range(self.layers):
            x = self.res_conv(x)

        x += identity
        x = self.conv_expand(x)
        x = self.mfm1(x)

        x = self.conv3x3(x)
        x = self.mfm2(x)
        return x


def lcnn4(in_channels=1, num_classes=1, out_bias=False, fc_dropout=0.7):
    """
    LightCNN-4 as described in the original LCNN paper.
    """
    return nn.Sequential(
        # Layer 1
        MFMConv2D(in_channels=in_channels, out_channels=48, kernel_size=9),
        nn.MaxPool2d(2),
        # Layer 2
        MFMConv2D(in_channels=48, out_channels=96, kernel_size=5),
        nn.MaxPool2d(2),
        # Layer 3
        MFMConv2D(in_channels=96, out_channels=128, kernel_size=5),
        nn.MaxPool2d(2),
        # Layer 4
        MFMConv2D(in_channels=128, out_channels=192, kernel_size=3),
        nn.MaxPool2d(2),
        # GAP
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        # FC
        nn.Linear(192, 512),
        nn.Dropout(p=fc_dropout),
        MaxFeatureMap21(512),
        nn.Linear(256, num_classes, bias=out_bias))


def lcnn9(in_channels=1, num_classes=1, out_bias=False, fc_dropout=0.7):
    """
    LightCNN-9 as described in the original LCNN paper.
    """
    return nn.Sequential(
        # Layer 1
        MFMConv2D(in_channels=in_channels, out_channels=48, kernel_size=5),
        nn.MaxPool2d(2),
        # Layer 2-3
        MFMConv2D(in_channels=48, out_channels=48, kernel_size=1),
        MFMConv2D(in_channels=48, out_channels=96, kernel_size=3),
        nn.MaxPool2d(2),
        # Layer 4-5
        MFMConv2D(in_channels=96, out_channels=96, kernel_size=1),
        MFMConv2D(in_channels=96, out_channels=192, kernel_size=3),
        nn.MaxPool2d(2),
        # Layer 6-7
        MFMConv2D(in_channels=192, out_channels=192, kernel_size=1),
        MFMConv2D(in_channels=192, out_channels=128, kernel_size=3),
        # Layer 8-9
        MFMConv2D(in_channels=128, out_channels=128, kernel_size=1),
        MFMConv2D(in_channels=128, out_channels=128, kernel_size=3),
        nn.MaxPool2d(2),
        # GAP
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        # FC
        nn.Linear(128, 512),
        nn.Dropout(p=fc_dropout),
        MaxFeatureMap21(512),
        nn.Linear(256, num_classes, bias=out_bias))


def lcnn29(in_channels=1, num_classes=1, out_bias=False, fc_dropout=0.7):
    """
    LightCNN-29 as described in the original LCNN paper.
    """
    return nn.Sequential(
        # Layer 1
        MFMConv2D(in_channels=in_channels, out_channels=48, kernel_size=5),
        nn.MaxPool2d(2),
        # Layer 2-5
        MFMResBlock(layers=1, in_channels=48, out_channels=96),
        nn.MaxPool2d(2),
        # Layer 6-11
        MFMResBlock(layers=2, in_channels=96, out_channels=192),
        nn.MaxPool2d(2),
        # Layer 12-29
        MFMResBlock(layers=3, in_channels=192, out_channels=128),
        MFMResBlock(layers=4, in_channels=128, out_channels=128),
        nn.MaxPool2d(2),
        # GAP
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        # FC
        nn.Linear(128, 512),
        nn.Dropout(p=fc_dropout),
        MaxFeatureMap21(512),
        nn.Linear(256, num_classes, bias=out_bias))


def lcnnSTC(in_channels=1, num_classes=1, out_bias=False, fc_dropout=0.75):
    """
    LCNN architecture used in
    'STC Antispoofing Systems for the ASVspoof2019 challenge', Lavrentyeva et al.
    arXiv:1904.05576
    """
    return nn.Sequential(
        MFMConv2D(in_channels=in_channels, out_channels=32, kernel_size=5),
        #
        nn.MaxPool2d(2),
        #
        MFMConv2D(in_channels=32, out_channels=32, kernel_size=1),
        nn.BatchNorm2d(32),
        MFMConv2D(in_channels=32, out_channels=48),
        #
        nn.MaxPool2d(2),
        nn.BatchNorm2d(48),
        #
        MFMConv2D(in_channels=48, out_channels=48, kernel_size=1),
        nn.BatchNorm2d(48),
        MFMConv2D(in_channels=48, out_channels=64),
        #
        nn.MaxPool2d(2),
        #
        MFMConv2D(in_channels=64, out_channels=64, kernel_size=1),
        nn.BatchNorm2d(64),
        MFMConv2D(in_channels=64, out_channels=32),
        nn.BatchNorm2d(32),
        MFMConv2D(in_channels=32, out_channels=32, kernel_size=1),
        nn.BatchNorm2d(32),
        MFMConv2D(in_channels=32, out_channels=32),
        #
        nn.MaxPool2d(2),
        #
        nn.AdaptiveAvgPool2d((1, 1)),  # Originally, Flatten()
        nn.Flatten(),
        #
        nn.Linear(32, 160),
        nn.Dropout(fc_dropout),
        MaxFeatureMap21(160),
        nn.BatchNorm1d(80),
        nn.Linear(80, num_classes, bias=out_bias))