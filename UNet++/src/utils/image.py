#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Jun 20 19:20:13 2022

@author: Nacriema

Refs:

"""
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import utils


def resize(img, size, keep_aspect_ratio=True, resample=Image.ANTIALIAS):
    # img.size return (W, H)
    # In the current task, image we use is 16:9 ratio (W:H) = (16:9)
    # Some suggestion size are: (1280, 720) and (1920, 1080)

    # TODO: XXX BUGs here !!!
    if isinstance(size, (int, float)):
        assert keep_aspect_ratio
        ratio = float(np.sqrt(size/(img.size[0] * img.size[1])))
        size = round(ratio * img.size[0]), round(ratio * img.size[1])

    elif keep_aspect_ratio:
        ratio = float(min([s1 / s2 for s1, s2 in zip(size, img.size)]))  # XXX bug with np.float64 and round
        size = round(ratio * img.size[0]), round(ratio * img.size[1])

    return img.resize(size, resample=resample)


def show_batch(sample_batched):
    """Show image and the ground Truth of a batch of samples."""
    images_batch, ground_batch = sample_batched[0], sample_batched[1]
    batch_size = len(images_batch)

    print(f"Image batch size: {images_batch.shape}")
    print(f"Ground Truth batch size: {ground_batch.shape}")

    processed = torch.unsqueeze(ground_batch, dim=1)
    print(f"Ground Truth batch size after: {processed.shape}")
    print(f"Ground Truth values: {torch.unique(processed)}")
    # Create image
    fig, ax = plt.subplots(nrows=3)

    image_grid = utils.make_grid(images_batch, nrow=batch_size)
    ax[0].imshow(image_grid.numpy().transpose((1, 2, 0)))
    ax[0].set_title("Image batch from dataloader")

    ground_grid = utils.make_grid(torch.unsqueeze(ground_batch, dim=1), nrow=batch_size)
    ax[1].imshow(ground_grid.numpy().transpose((1, 2, 0)))
    ax[1].set_title("Ground truth batch from dataloader")
