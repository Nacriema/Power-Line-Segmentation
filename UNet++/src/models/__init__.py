#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Jun 20 19:45:42 2022

@author: Nacriema

Refs:

"""
from functools import partial
from .unet_pp import NestedUnet


def get_model(name=None):
    if name is None:
        name = "resnet34-unet++"
    return {
        'resnet34-unet++': partial(NestedUnet, encoder_name='resnet34')
    }[name]

