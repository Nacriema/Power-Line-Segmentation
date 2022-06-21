#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Jun 20 19:50:27 2022

@author: Nacriema

Refs:

"""
from functools import partial
from torch import nn
from collections import OrderedDict
import torch
import numpy as np


def get_norm_layer(**kwargs):
    name = kwargs.get('name', 'batch_norm')
    momentum = kwargs.get('momentum', 0.1)
    affine = kwargs.get('affine', True)
    track_stats = kwargs.get('track_running_stat', True)
    num_groups = kwargs.get('num_groups', 32)

    norm_layer = {
        'batch_norm': partial(nn.BatchNorm2d, momentum=momentum, affine=affine, track_running_stats=track_stats),
        'group_norm': partial(nn.GroupNorm, num_groups=num_groups, affine=affine),
        'instance_norm': partial(nn.InstanceNorm2d, momentum=momentum, affine=affine, track_running_stats=track_stats),
    }[name]
    if norm_layer.func == nn.GroupNorm:
        return lambda num_channels: norm_layer(num_channels=num_channels)
    else:
        return norm_layer


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def safe_model_state_dict(state_dict):
    """Convert a state dict saved from a DataParallel module to normal module state_dict."""
    if not next(iter(state_dict)).startswith("module."):
        return state_dict  # abort if dict is not a DataParallel model_state
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k[7:]] = v   # remove 'module.' prefix
    return new_state_dict


# Early stopping pytorch
# Link: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        :param patience: How long to wait after last time validation loss improved
        :param verbose: If True, print a message for each validation loss improvement
        :param delta: Minimum change in the monitored quantify as an improvement
        :param path: Path for the checkpoint to be saved to
        :param trace_func: trace print function.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, state_dict):
        # val_loss is > 0  -> score < 0
        # val_loss_better < val_loss_bad -> score_better > score_bad
        score = - val_loss  # XXX: Consider this line !!
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, state_dict)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.save_checkpoint(val_loss, state_dict)
            self.counter = 0

    def save_checkpoint(self, val_loss, state_dict):
        """Save model when validation loss decrease."""
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min: .6f} ---> {val_loss: .6f}). Saving model '
                            f'...')

        torch.save(state_dict, self.path)
        self.val_loss_min = val_loss
