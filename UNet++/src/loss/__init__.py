#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Jun 18 18:23:33 2022

@author: Nacriema

Refs:

"""
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss


def get_loss(name):
    if name is None:
        name = 'bce'
    return {
        'bce': BCEWithLogitsLoss,
        'cross_entropy': CrossEntropyLoss,
    }[name]
