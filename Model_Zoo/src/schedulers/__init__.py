#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Jun 18 17:03:23 2022

@author: Nacriema

Refs:

"""
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, MultiStepLR


def get_scheduler(name):
    if name is None:
        name = 'constant_lr'
    return {
        "multi_step": MultiStepLR,
        "cosine_annealing": CosineAnnealingLR,
        "exp_lr": ExponentialLR,
    }[name]
