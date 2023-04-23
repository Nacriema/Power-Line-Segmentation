#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Jun 17 20:54:05 2022

@author: Nacriema

Refs:

"""

from PIL import Image
import numpy as np

# Test whent transposed with the image
# img = Image.open('27_00740.jpg')
# img = img.transpose(Image.ROTATE_180)
# img.show()

# I = np.array(img)
# print(I.shape)
# print(I.mean(axis=(0, 1)).shape)
# print(np.unique(I))

# arr_segmap = np.array(Image.open('../../docs/images/27_00740_Labels.png'), dtype=np.uint8)
# print(arr_segmap.shape)  # H x W x C

# from constant import LABEL_TO_COLOR_MAPPING
# restricted_labels = [1]
#
# print(LABEL_TO_COLOR_MAPPING)
# rs = [LABEL_TO_COLOR_MAPPING[l] for l in restricted_labels]
# print(rs)
