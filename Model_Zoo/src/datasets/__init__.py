#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Jun 20 19:38:28 2022

@author: Nacriema

Refs:

"""
from abc import ABC
from .segmentation import AbstractSegDataset


def get_dataset(dataset_name):
    class Dataset(AbstractSegDataset, ABC):
        name = dataset_name
    return Dataset
