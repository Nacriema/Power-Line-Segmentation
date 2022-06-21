#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Jun 20 19:56:11 2022

@author: Nacriema

Refs:

"""
import torch


def get_mean_std(loader):
    """Use this to get the mean and standard deviation of the given data"""
    # VAR[X] = E[X**2] - E[X]**2
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    # Loader is Iterable [<return type of __get_item__>], in my case is Iterable[List<Image_Batch, Label_Batch>]
    # Then for loop to get a batch
    for data, _ in loader:
        print(f"Data batch shape: {data.shape}")
        channels_sum += torch.mean(data, dim=[0, 2, 3])   # Calculate E[X]
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])   # Calculate E[X**2]
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean**2)**0.5  # STD = Sqrt(VAR)

    return mean, std
