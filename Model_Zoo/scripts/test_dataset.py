#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Jun 18 11:21:02 2022

@author: Nacriema

Refs:

"""
# import sys
#
# import yaml
# from torch.utils.data import DataLoader
# from datasets import get_dataset
# from utils.image import show_batch
# from datasets.tools import get_mean_std
# import matplotlib.pyplot as plt
# import os
# import torch

# # Load config by yaml
#
# with open(os.path.join(sys.path[0], 'test.yml')) as fp:
#     cfg = yaml.load(fp, Loader=yaml.FullLoader)
#
# print(cfg["dataset"])
# dataset_cfg = cfg["dataset"]
#
# batch_size = 5

# train_dataset = get_dataset(dataset_name=dataset_cfg['name'])(split="train", **dataset_cfg)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
#
# first = True
#
# for i_batch, sample_batched in enumerate(train_loader):
#     print(i_batch, sample_batched[0].size(), sample_batched[1].size())
#
#     # Observe the 1 st batch and stop.
#     if i_batch == 0:
#         show_batch(sample_batched)
#         plt.axis("off")
#         plt.ioff()
#         plt.show()
#         print(sample_batched[1].size())
#         break
#     # Calculating labels weights for weighted loss computation
#     print(sample_batched[1].dtype)
#     if first:
#         first = False
#         result = torch.unique(sample_batched[1], return_counts=True)[1]
#     result += torch.unique(sample_batched[1], return_counts=True)[1]
#
# print(f"RESULT: {result}")
# result = 1. / result
# print(f"WEIGHT: {result}")
# print("======================")



# for data, _ in train_loader:
#     print(data.shape)

# mean, std = get_mean_std(train_loader)
# print(f"Mean: {mean}, Std: {std}")
#


