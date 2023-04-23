#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Jun 20 19:33:23 2022

@author: Nacriema

Refs:

"""
from pathlib import Path

# Project and source files
PROJECT_PATH = Path(__file__).parent.parent.parent  # UNet++
MODELS_PATH = PROJECT_PATH / 'models'
DATASET_PATH = PROJECT_PATH / 'datasets'
CONFIGS_PATH = PROJECT_PATH / 'configs'

MODEL_FILE = 'model.pkl'
# print(PROJECT_PATH)
