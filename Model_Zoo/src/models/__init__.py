#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Jun 20 19:45:42 2022

@author: Nacriema

Refs:

"""
from functools import partial
import torch
from .unet_pp import NestedUnet
from ..utils import coerce_to_path_and_check_exist
from .tools import safe_model_state_dict


def get_model(name=None):
    if name is None:
        name = "resnet34-unet++"
    return {
        'resnet34-unet++': partial(NestedUnet, encoder_name='resnet34')
    }[name]


def load_model_from_path(model_path, device=None, attributes_to_return=None, eval_mode=True):
    """This function is use when training phase finish, reload model for testing phase"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(coerce_to_path_and_check_exist(model_path), map_location=device.type)

    # Ok, Trick is here, change pretrained_encoder value to false for better initializing the model.
    checkpoint['model_kwargs']['pretrained_encoder'] = False
    model = get_model(checkpoint["model_name"])(checkpoint["n_classes"], **checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(safe_model_state_dict(checkpoint["model_state"]))

    if eval_mode:
        model.eval()
    if attributes_to_return is not None:
        if isinstance(attributes_to_return, str):
            attributes_to_return = [attributes_to_return]
        return model, [checkpoint.get(key) for key in attributes_to_return]
    else:
        return model
