# Residual Networks adapted from Torchvision's implementation
# https://github.com/pytorch/vision/blob/020513dc9f03c62d7f92c93b8f0f10a41a5768b1/torchvision/models/resnet.py
# Introducing squeeze-excitation into ResBlocks
# and ResNeWt, as proposed in the following paper: https://doi.org/10.1109/APSIPAASC47483.2019.9023158
# which applies ResNeXt-like strategy to the 2-layer ResNet architectures (ResNet18 and 34)

from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor

from .se_module import SELayer


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        reduction: int = None,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        self.use_se = True if reduction else False
        # self.se = SELayer(...) if self.reduction else lambda x: x
        # avoid using lambda functions as it negatively impacts code readability
        self.squeeze_excitation = SELayer(planes, reduction=reduction) if self.use_se else None

    def _forward_conv(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        return out

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self._forward_conv(x)

        if self.use_se:
            out = self.squeeze_excitation(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        reduction: int = None,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(in_planes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)  # Why not just use "in_planes" as out filters ?
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.use_se = True if reduction else False
        # self.se = SELayer(...) if self.reduction else lambda x: x
        # avoid using lambda functions as it negatively impacts code readability
        self.squeeze_excitation = SELayer(planes * self.expansion, reduction=reduction) if self.use_se else None

    def _forward_conv_block(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        return out

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self._forward_conv_block(x)

        if self.use_se:
            out = self.squeeze_excitation(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNeWtBlock(nn.Module):
    expansion: int = 2

    def __init__(
        self,
        in_planes: int,
        planes: int,
        reduction: int = None,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 32,
        base_width: int = 4,  # == width per group
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv3x3(in_planes, self.width, stride)
        self.bn1 = norm_layer(self.width)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(self.width, self.width, groups=groups)
        self.bn2 = norm_layer(self.width)
        self.downsample = downsample

        self.use_se = True if reduction else False
        # self.se = SELayer(...) if self.reduction else lambda x: x
        # avoid using lambda functions as it negatively impacts code readability
        self.squeeze_excitation = SELayer(self.width, reduction=reduction) if self.use_se else None

    def _forward_conv(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        return out

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self._forward_conv(x)

        if self.use_se:
            out = self.squeeze_excitation(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# ---------------- ResNet class --------------------
class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck, ResNeWtBlock]],
        layers: List[int],
        reduction: Union[int, List[int]] = None,
        input_channels: int = 1,
        num_classes: int = 1000,
        out_bias: bool = True,
        fc_dropout: Optional[float] = None,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        if reduction is None:
            self.reductions = [None] * len(layers)
        elif isinstance(reduction, int):
            assert reduction >= 1
            self.reductions = [reduction] * len(layers)
        elif isinstance(reduction, list):
            assert all(isinstance(r, int) and r >= 1 for r in reduction)
            self.reductions = reduction
        else:
            raise AssertionError("Reduction has to be an integer or list of integers of same length as 'layers'")

        self.in_planes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(input_channels, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], reduction=self.reductions[0])
        self.layer2 = self._make_layer(block, 128, layers[1], reduction=self.reductions[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], reduction=self.reductions[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], reduction=self.reductions[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # this is a GAP layer (avg pooling with output WxH = 1x1)

        self.use_dropout = True if fc_dropout is not None else False
        self.dropout = nn.Dropout(fc_dropout) if self.use_dropout else None

        self.fc = nn.Linear(512 * block.expansion, num_classes, bias=out_bias)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif (isinstance(m, BasicBlock) or isinstance(m, ResNeWtBlock)
                      ) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck, ResNeWtBlock]],
        planes: int,
        blocks: int,
        reduction: Union[int, None],
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_planes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = list()
        layers.append(
            block(
                in_planes=self.in_planes,
                planes=planes,
                stride=stride,
                reduction=reduction,
                downsample=downsample,
                groups=self.groups,
                base_width=self.base_width,
                dilation=previous_dilation,
                norm_layer=norm_layer
            )
        )
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    in_planes=self.in_planes,
                    planes=planes,
                    reduction=reduction,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def resnet18(num_classes, *args, **kwargs) -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, *args, **kwargs)


def resnet34(num_classes, *args, **kwargs) -> ResNet:
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, *args, **kwargs)


def resnet50(num_classes, *args, **kwargs) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, *args, **kwargs)


def resnet101(num_classes, *args, **kwargs) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, *args, **kwargs)


def resnet152(num_classes, *args, **kwargs) -> ResNet:
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, *args, **kwargs)


def se_resnet18(num_classes, *args, **kwargs) -> ResNet:
    return resnet18(num_classes, reduction=16, *args, **kwargs)


def se_resnet34(num_classes, *args, **kwargs) -> ResNet:
    return resnet34(num_classes, reduction=16, *args, **kwargs)


def se_resnet50(num_classes, *args, **kwargs) -> ResNet:
    return resnet50(num_classes, reduction=16, *args, **kwargs)


def se_resnet101(num_classes, *args, **kwargs) -> ResNet:
    return resnet101(num_classes, reduction=16, *args, **kwargs)


def se_resnet152(num_classes, *args, **kwargs) -> ResNet:
    return resnet152(num_classes, reduction=16, *args, **kwargs)


def resnewt18_32x4d(num_classes, *args, **kwargs) -> ResNet:
    kwargs["width_per_group"] = 4
    kwargs["groups"] = 32
    return ResNet(ResNeWtBlock, [2, 2, 2, 2], num_classes=num_classes, *args, **kwargs)


def resnewt34_32x4d(num_classes, *args, **kwargs) -> ResNet:
    kwargs["width_per_group"] = 4
    kwargs["groups"] = 32
    return ResNet(ResNeWtBlock, [3, 4, 6, 3], num_classes=num_classes, *args, **kwargs)


def se_resnewt18_32x4d(num_classes, *args, **kwargs) -> ResNet:
    kwargs["width_per_group"] = 4
    kwargs["groups"] = 32
    return resnewt18_32x4d(num_classes, reduction=16, *args, **kwargs)


def se_resnewt34_32x4d(num_classes, *args, **kwargs) -> ResNet:
    kwargs["width_per_group"] = 4
    kwargs["groups"] = 32
    return resnewt34_32x4d(num_classes, reduction=16, *args, **kwargs)


def resnext50_32x4d(num_classes, *args, **kwargs) -> ResNet:
    kwargs["width_per_group"] = 4
    kwargs["groups"] = 32
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, *args, **kwargs)


def resnext101_32x8d(num_classes, *args, **kwargs) -> ResNet:
    kwargs["width_per_group"] = 8
    kwargs["groups"] = 32
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, *args, **kwargs)


def resnext101_64x8d(num_classes, *args, **kwargs) -> ResNet:
    kwargs["width_per_group"] = 8
    kwargs["groups"] = 64
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, *args, **kwargs)


def se_resnext50_32x4d(num_classes, *args, **kwargs) -> ResNet:
    kwargs["width_per_group"] = 4
    kwargs["groups"] = 32
    return resnext50_32x4d(num_classes, reduction=16, *args, **kwargs)


def se_resnext101_32x8d(num_classes, *args, **kwargs) -> ResNet:
    kwargs["width_per_group"] = 8
    kwargs["groups"] = 32
    return resnext101_32x8d(num_classes, reduction=16, *args, **kwargs)


def se_resnext101_64x8d(num_classes, *args, **kwargs) -> ResNet:
    kwargs["width_per_group"] = 8
    kwargs["groups"] = 64
    return resnext101_64x8d(num_classes, reduction=16, *args, **kwargs)
