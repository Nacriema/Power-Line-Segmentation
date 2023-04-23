#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Jun 20 19:48:23 2022

@author: Nacriema

Refs:

Implementation of Unet++ with Resnet-34 as backbone
In the paper - TTPLA GAN: They use the model
"""
import torch
from torch import Tensor
from torch import nn
from typing import Any, List
from .resnet import get_resnet_model, BasicBlock, conv1x1, UpsampleConv, Upsample
from .tools import get_norm_layer
from ..utils.path import PROJECT_PATH
import os


os.environ['TORCH_HOME'] = str(PROJECT_PATH)


class NestedUnet(nn.Module):
    """UNet++ with Resnet-34 backbone"""
    @property
    def name(self):
        return self.enc_name + "-unet++"

    def __init__(self, num_classes: int, deep_supervision: bool = False, **kwargs: Any):
        super(NestedUnet, self).__init__()
        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(2, 2)

        # Simple up sample with no learnable parameter, this is used "inside" the model
        # TODO: XXX Bugs when image size not divided by 32, so I decided to define an Upsample module instead
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up = Upsample(mode='bilinear', align_corners=True)

        # Get the Norm layer and configs from the yaml file
        self.norm_layer_kwargs = kwargs.pop('norm_layer', dict())
        self.norm_layer = get_norm_layer(**self.norm_layer_kwargs)

        # Get the pretrained weights for resnet to make the model backbone
        self.enc_name = kwargs.get('encoder_name', 'resnet34')
        pretrained = kwargs.get('pretrained_encoder', True)
        resnet = get_resnet_model(self.enc_name)(pretrained=pretrained, progress=False)

        # Build the net structure based on the component of resnet above !
        # I applied the First conv and pooling of ResNet as the first conv
        self.conv0_0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)

        # Use the rest of Block to build the bone for the model, go deeper !
        self.conv1_0, self.conv2_0, self.conv3_0, self.conv4_0 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        # With resnet-34 this is 64, 64, 128, 256, 512
        nb_filter = [self.get_nb_out_channels(self.conv0_0),
                     self.get_nb_out_channels(self.conv1_0),
                     self.get_nb_out_channels(self.conv2_0),
                     self.get_nb_out_channels(self.conv3_0),
                     self.get_nb_out_channels(self.conv4_0)]

        # Reuse the BasicBlock of ResNet to build so called a Constant block (it just changed the features map)
        self.conv0_1 = self._create_residual_block(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv1_1 = self._create_residual_block(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv2_1 = self._create_residual_block(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv3_1 = self._create_residual_block(nb_filter[3] + nb_filter[4], nb_filter[3])

        self.conv0_2 = self._create_residual_block(nb_filter[0]*2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = self._create_residual_block(nb_filter[1]*2 + nb_filter[2], nb_filter[1])
        self.conv2_2 = self._create_residual_block(nb_filter[2]*2 + nb_filter[3], nb_filter[2])

        self.conv0_3 = self._create_residual_block(nb_filter[0]*3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = self._create_residual_block(nb_filter[1]*3 + nb_filter[2], nb_filter[1])

        self.conv0_4 = self._create_residual_block(nb_filter[0]*4 + nb_filter[1], nb_filter[0])

        if self.deep_supervision:
            # Try to modify something here !!!
            self.final1 = self._upsampling_layer(in_channels=nb_filter[0], out_channels=num_classes)
            self.final2 = self._upsampling_layer(in_channels=nb_filter[0], out_channels=num_classes)
            self.final3 = self._upsampling_layer(in_channels=nb_filter[0], out_channels=num_classes)
            self.final4 = self._upsampling_layer(in_channels=nb_filter[0], out_channels=num_classes)

        else:
            self.final = self._upsampling_layer(in_channels=nb_filter[0], out_channels=num_classes)

    def forward(self, x: Tensor) -> List[Any]:
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(x0_0)
        x0_1 = self.conv0_1(torch.cat([x0_0, x1_0], 1))

        x2_0 = self.conv2_0(x1_0)
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0, output_size=x1_0.shape)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, x1_1], 1))

        x3_0 = self.conv3_0(x2_0)
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0, output_size=x2_0.shape)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1, output_size=x1_0.shape)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, x1_2], 1))

        x4_0 = self.conv4_0(x3_0)
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0, output_size=x3_0.shape)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1, output_size=x2_0.shape)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2, output_size=x1_0.shape)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, x1_3], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1, output_size=x.shape)
            output2 = self.final2(x0_2, output_size=x.shape)
            output3 = self.final3(x0_3, output_size=x.shape)
            output4 = self.final4(x0_4, output_size=x.shape)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4, output_size=x.shape)
            return output

    def _upsampling_layer(self, in_channels, out_channels, use_conv1x1=True):
        return UpsampleConv(in_channels, out_channels, use_conv1x1=use_conv1x1)

    def get_nb_out_channels(self, layer: nn.Sequential) -> int:
        return list(filter(lambda e: isinstance(e, nn.Conv2d), layer.modules()))[-1].out_channels

    def _create_residual_block(self, inplanes, planes, stride=1):
        downsample = self._create_downsample(inplanes=inplanes, planes=planes, stride=stride)
        layer = BasicBlock(inplanes=inplanes, planes=planes, downsample=downsample, stride=stride)

        # Initializing weights
        for m in layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        return layer

    def _create_downsample(self, inplanes, planes, stride):
        return nn.Sequential(
            conv1x1(inplanes, planes, stride),
            # Use Norm layer config to instantiate Norm layer instead
            # nn.BatchNorm2d(planes)
            self.norm_layer(planes)
        )

# OK DONE THE MODEL PART  !!!

# net = NestedUnet(num_classes=1)
# net = get_model(name="resnet34-unet++")(num_classes=1)
# print(f'Total trainable params: {count_parameters(net): ,}')

# net._create_residual_block(inplanes=3, planes=10)
# print(net.name)
# downsample = net.create_downsample(inplanes=3, planes=10, stride=1)
# layer = BasicBlock(inplanes=3, planes=10, downsample=downsample, stride=1)
#
# inp = torch.randn((5, 3, 200, 200))
# out = layer(inp)
# print("xxxxx")
# print(out.shape)
#
# inp = torch.randn((1, 3, 720, 1280))
# out = net(inp)
# print(out.shape)
# print("xxxxxxx")