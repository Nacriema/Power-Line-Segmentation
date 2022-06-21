#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Jun 16 12:17:39 2022

@author: Nacriema

Refs:

Making resnet basic block

Tutorial:

* https://www.analyticsvidhya.com/blog/2021/08/all-you-need-to-know-about-skip-connections/
* https://github.com/Nacriema/PMIS-Tool/blob/985018c425e1c9446610a83a38e803dd822775dc/vegseg/models/resnet.py
* https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

Use PEP 484 Type Hints when coding
Currently, I just use ResNet 34
"""
import torch
# from toolz import keyfilter
from torch import nn
from torch import Tensor
from typing import Optional, Callable, Type, Union, List, Any

from torch.utils.model_zoo import load_url as load_state_dict_from_url


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding
    With the default params, this 3x3 conv will return the output size same as input size
    Due to: o = [(i - k + 2p)/s] + 1 = [(i - 3 + 2)/1] + 1 = i
    If stride=2 then:
        o = [(i-k + 2p)/s] + 1 = [(i - 3 + 2)/2] + 1 =
    """
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
    """1x1 convolution
    By default, this convolution will remain the size of input
    o = [(i - k + 2p)/s] + 1 = [(i - 1 + 0)/1] + 1 = i
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class UpsampleCatConv(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=None, mode='bilinear', use_conv1x1=False):
        super().__init__()
        norm_layer = norm_layer if norm_layer is not None else nn.BatchNorm2d
        conv_layer = conv1x1 if use_conv1x1 else conv3x3
        self.mode = mode
        self.conv = conv_layer(in_channels, out_channels)
        self.norm = norm_layer(out_channels)
        self.act = nn.ReLU()

    def forward(self, x, other):
        x = nn.functional.interpolate(x, size=(other.size(2), other.size(3)), mode=self.mode, align_corners=False)
        x = torch.cat((x, other), dim=1)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class UpsampleConv(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=None, mode='bilinear', use_conv1x1=False):
        super(UpsampleConv, self).__init__()
        norm_layer = norm_layer if norm_layer is not None else nn.BatchNorm2d
        conv_layer = conv1x1 if use_conv1x1 else conv3x3
        self.mode = mode
        self.conv = conv_layer(in_channels, out_channels)
        self.norm = norm_layer(out_channels)
        self.act = nn.ReLU()

    def forward(self, x, output_size):
        x = nn.functional.interpolate(x, size=output_size[2:], mode=self.mode, align_corners=False)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class Upsample(nn.Module):
    def __init__(self, mode='bilinear', align_corners=True):
        super(Upsample, self).__init__()
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x, output_size):
        x = nn.functional.interpolate(x, size=output_size[2:], mode=self.mode, align_corners=self.align_corners)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 ) -> None:
        """A basic residual block of ResNet-this is used in resnet18, resnet34, in deeper resnet,use Bottleneck instead
        Checkout the shape between input vs output after this block
        """
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(BottleNeck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block: Type[Union[BasicBlock, BottleNeck]],
                 layers: List[int],
                 num_classes: int = 1000,
                 zero_init_residual: bool = False,
                 groups: int = 1,
                 with_per_group: int = 64,
                 replace_stride_with_dilation: Optional[List[bool]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super(ResNet, self).__init__()
        # self.norm_layer_kwargs = kwargs.get('norm_layer', dict())
        # norm_layer = get_norm_layer(**self.norm_layer_kwargs)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None"
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = with_per_group

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        # TODO: MaxPool2d with kernel_size, stride, padding !!!
        # out = [(in + 2p - d*(k-1)-1)/s] + 1 = [(in + 2 - (3-1)-1)/2] + 1 = [(in - 1)/2] + 1
        # This is applied for both odd and even case of input (I think):
        # 10 -> 5, 9 -> 5
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block=block, planes=64, blocks=layers[0])
        self.layer2 = self._make_layer(block=block, planes=128, blocks=layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block=block, planes=256, blocks=layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block=block, planes=512, blocks=layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # TODO: This component should be learned !!!
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleNeck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)
                if isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self,
                    block: Type[Union[BasicBlock]],
                    planes: int,
                    blocks: int,
                    stride: int = 1,
                    dilate: bool = False) -> nn.Sequential:
        """

        :param block:
        :param planes:
        :param blocks: This is the block size, in resnet34 this is [3, 4, 6, 3]
        :param stride:
        :param dilate:
        :return:
        """
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        # TODO: By default, we not use dilate
        if dilate:
            self.dilation *= stride
            stride = 1
        # TODO: By default, stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            # Down sample is used to adjust the number of features map as well as the feature map size
            # between identity vs the out before the adding step in BasicBlock
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        # The fist layer of each block take the stride param from _make_layer function to reduce the size of input
        layers = [
            block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation,
                  norm_layer)
        ]

        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        # TODO: Learn this ! _forward_impl
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


model_urls = {
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
}


def _resnet(arch: str,
            block: Type[Union[BasicBlock, BottleNeck]],
            layers: List[int],
            pretrained: bool,
            progress: bool,
            **kwargs: Any) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


# Public methods to get multiple models
def resnet34(pretrained: bool = False,
             progress: bool = True,
             **kwargs: Any) -> ResNet:
    """ResNet-34 model"""
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)


def get_resnet_model(name: str = None) -> ResNet:
    if name is None:
        name = 'resnet34'
    return {
        'resnet34': resnet34
    }[name]


# Test the basic block OK DONE this part

# STRIDE = 1
# downsample = nn.Sequential(
#     conv1x1(3, 10, stride=STRIDE),
# )
# layer = BasicBlock(inplanes=3, planes=10, downsample=downsample, stride=STRIDE)
# inp = torch.randn((5, 3, 200, 200), requires_grad=True)
# out = layer(inp)
# print("====")
# print(out.shape)
