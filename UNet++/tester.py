#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Jun 17 18:57:50 2022

@author: Nacriema

Refs:

"""
import argparse
from PIL import Image
import yaml

import torch

from datasets import get_dataset
from utils import coerce_to_path_and_check_exist
from utils.constant import MODEL_FILE


class Tester:
    """Pipeline to test"""
    def __init__(self):
        pass
