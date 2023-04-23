#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Jun 20 19:40:16 2022

@author: Nacriema

Refs:

DataLoader part for the dataset, code will be the same idea with docExtractor
"""
from abc import ABCMeta, abstractmethod
from typing import Any
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import torch
from torch.utils.data.dataset import Dataset as TorchDataset
from numpy.random import choice, uniform, randint

# TODO: Hummm, A BIG PROBLEM HERE ! MAYBE NOT ?
from ..utils import coerce_to_path_and_check_exist, get_files_from_dir
from ..utils.image import resize
from ..utils.constant import BACKGROUND_LABEL, BACKGROUND_COLOR, SEG_GROUND_TRUTH_FMT, LABEL_TO_COLOR_MAPPING
from ..utils.path import DATASET_PATH

INPUT_EXTENSIONS = ['jpeg', 'jpg', 'JPG']
LABEL_EXTENSIONS = 'png'

# Data augmentations
BLUR_RADIUS_RANGE = (0, 0.5)
BRIGHTNESS_FACTOR_RANGE = (0.9, 1.1)
CONTRAST_FACTOR_RANGE = (0.5, 1.5)
ROTATION_ANGLE_RANGE = (-10, 10)
SAMPLING_RATIO_RANGE = (0.6, 1.4)
TRANSPOSITION_CHOICES = [None, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]

# TODO: XXX BUGs here make the BATCH not available
TRANSPOSITION_WEIGHTS = [0.5, 0., 0.5, 0.]


class AbstractSegDataset(TorchDataset):
    """Abstract torch dataset for segmentation task."""
    __metaclass__ = ABCMeta

    # Ref: https://stackoverflow.com/questions/5960337/how-to-create-abstract-properties-in-python-abstract-classes
    @property
    @abstractmethod
    def name(self):
        pass

    @property
    def root_path(self):
        return DATASET_PATH / self.name

    def __init__(self, split: str, restricted_labels=None, **kwargs):
        """
        :param split: partition of dataset (train, test or val)
        :param restricted_labels:
        :param kwargs:
        """
        self.data_path = coerce_to_path_and_check_exist(self.root_path) / split
        self.split = split
        self.input_files, self.label_files = self._get_input_label_files()

        self.size = len(self.input_files)
        self.restricted_labels = sorted(restricted_labels)

        self.restricted_colors = [LABEL_TO_COLOR_MAPPING[l] for l in self.restricted_labels]

        self.label_idx_color_mapping = {self.restricted_labels.index(l) + 1: c for l, c in zip(self.restricted_labels, self.restricted_colors)}  # Ex: {1: (255, 255, 255)}
        self.color_label_idx_mapping = {c: l for l, c in self.label_idx_color_mapping.items()}  # Ex: {(255, 255, 255): 1}
        self.fill_background = BACKGROUND_LABEL in self.restricted_labels

        # TODO: Consider this !
        self.n_classes = len(self.restricted_labels) + 1
        self.img_size = kwargs.get('img_size')
        self.keep_aspect_ratio = kwargs.get('keep_aspect_ratio', True)

        # TODO: ?!
        self.base_line_dilation_iter = kwargs.get('baseline_dilation_iter', 1)

        # self.normalize = kwargs.get('normalize', True)
        # TODO: Instead of read from arg pass, read from **kwargs of the yaml config file
        self.normalize = kwargs.get('data_distribution', None)
        self.data_augmentation = kwargs.get('data_augmentation', True) and split == 'train'

        # PARAMS when doing transformation
        self.blur_radius_range = kwargs.get('blur_radius_range', BLUR_RADIUS_RANGE)
        self.brightness_factor_range = kwargs.get('brightness_factor_range', BRIGHTNESS_FACTOR_RANGE)
        self.contrast_factor_range = kwargs.get('contrast_factor_range', CONTRAST_FACTOR_RANGE)
        self.rotation_angle_range = kwargs.get('rotation_angle_range', ROTATION_ANGLE_RANGE)
        self.sampling_ratio_range = kwargs.get('sampling_ratio_range', SAMPLING_RATIO_RANGE)
        self.sampling_max_nb_pixels = kwargs.get('sampling_max_nb_pixels')
        self.transposition_weights = kwargs.get('transposition_weights', TRANSPOSITION_WEIGHTS)

    def _get_input_label_files(self):
        input_files = get_files_from_dir(str(self.data_path), INPUT_EXTENSIONS, sort=True)
        label_files = get_files_from_dir(str(self.data_path), [LABEL_EXTENSIONS])

        if len(label_files) == 0 and self.split == 'test':
            return input_files, None

        elif len(input_files) != len(label_files):
            raise RuntimeError("The number of inputs and labels do not match !!!")

        # Carefully check the input and label image pair if small dataset
        if len(input_files) < 1e5:
            inputs = [p.stem for p in input_files]
            labels = [str(p.name) for p in label_files]
            invalid = []

            for name in inputs:
                if SEG_GROUND_TRUTH_FMT.format(name, LABEL_EXTENSIONS) not in labels:
                    invalid.append(name)
            if len(invalid) > 0:
                raise FileNotFoundError("Some inputs don't have the corresponding labels: {}".format(' '. join(invalid)))

        else:
            assert len(input_files) == len(label_files)

        label_files = [path.parent / SEG_GROUND_TRUTH_FMT.format(path.stem, LABEL_EXTENSIONS) for path in input_files]
        return input_files, label_files

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if self.data_augmentation:
            augm_kwargs = {
                'blur_radius': uniform(*self.blur_radius_range),  # This will return a scalar sampled uniform in range
                'brightness': uniform(*self.brightness_factor_range),
                'contrast': uniform(*self.contrast_factor_range),
                'rotation': randint(*self.rotation_angle_range),
                'sampling_ratio': uniform(*self.sampling_ratio_range),
                'transpose': choice(TRANSPOSITION_CHOICES, p=self.transposition_weights)
            }
        else:
            augm_kwargs = {}

        # 1. Process the input image
        # Note that PIL convert to np -> value at uint8, so we need to convert it to float32 and scale 1.
        # Note that the np array read is in shape (height x width x channel)
        inp = np.array(self.transform(Image.open(self.input_files[idx]), **augm_kwargs), dtype=np.float32) / 255.

        # TODO: I think the normalize step must be done outsize, when the batch is loaded, so I disable it here !
        # if self.normalize:
        #     # NOTICE: Normalize through each color channel, this is Instance Norm, not Batch Norm
        #     inp = ((inp - inp.mean(axis=(0, 1))) / (inp.std(axis=(0, 1)) + 10**-7))

        # Instead use the dataset mean and std to normalize each instance in data
        if self.normalize is not None:
            inp = ((inp - np.array(self.normalize['mean'], dtype=np.float32)) / (np.array(self.normalize['std'], dtype=np.float32) + 10**-7))

        # Assure that the input image has 3 channels color
        inp = np.dstack([inp, inp, inp]) if len(inp.shape) == 2 else inp
        inp = torch.from_numpy(inp.transpose((2, 0, 1))).float()  # HWC -> CHW tensor

        # 2. Process the corresponding label image
        # Note that by default reading of PIL, it read a png image convert to np has shape (H, W, C)
        # When performing transform, because it is the label, so we not transform the color space by set the is_gt=True
        if self.label_files is None:
            label = None
        else:
            img = Image.open(self.label_files[idx])
            arr_segmap = np.array(self.transform(img, is_gt=True, **augm_kwargs), dtype=np.uint8)
            unique_colors = set([color for size, color in img.getcolors()]).difference({BACKGROUND_COLOR})
            label = self.encode_segmap(arr_segmap, unique_colors)

            # XXX: Add one extra dimension to the label such has shape [1, H, W]
            # This is not appropriate when use CrossEntropyLoss
            # label = torch.unsqueeze(label, dim=0)

        return inp, label

    def transform(self, img, is_gt: bool = False, **augm_kwargs: Any):
        if self.img_size is not None:

            # Resample param for resize
            resample = Image.NEAREST if is_gt else Image.ANTIALIAS

            # TODO: XXX BUGs here make the batch training not available INSPECTOR THIS !!!
            # Due to the random sampling ratio on each instance, then the batch concat is not available !!

            # if self.data_augmentation:
            #     size = tuple(map(lambda s: round(augm_kwargs['sampling_ratio'] * s), self.img_size))
            #     if self.sampling_max_nb_pixels is not None and self.keep_aspect_ratio:
            #         ratio = float(min([s1 / s2 for s1, s2 in zip(size, img.size)]))
            #         real_size = round(ratio * img.size[0]), round(ratio * img.size[1])
            #         nb_pixels = np.product(real_size)
            #         if nb_pixels > self.sampling_max_nb_pixels:
            #             ratio = float(np.sqrt(self.sampling_max_nb_pixels / nb_pixels))
            #             size = round(ratio * real_size[0]), round(ratio * real_size[1])
            # else:
            #     size = self.img_size

            size = self.img_size
            # CHECK THE SIZE OF IMAGE HERE

            img = resize(img, size=size, keep_aspect_ratio=self.keep_aspect_ratio, resample=resample)

        if self.data_augmentation:
            # 1. Augmentation the pixel space of image, this is not applied for ground truth label
            if not is_gt:
                img = img.filter(ImageFilter.GaussianBlur(radius=augm_kwargs['blur_radius']))
                img = ImageEnhance.Brightness(img).enhance(augm_kwargs['brightness'])
                img = ImageEnhance.Contrast(img).enhance(augm_kwargs['contrast'])

            # Resample param for rotate
            resample = Image.NEAREST if is_gt else Image.BICUBIC
            img = img.rotate(augm_kwargs['rotation'], resample=resample, fillcolor=BACKGROUND_LABEL)
            if augm_kwargs['transpose'] is not None:
                img = img.transpose(augm_kwargs['transpose'])

        return img

    def encode_segmap(self, arr_segmap, unique_colors=None):
        if unique_colors is None:
            # unique_colors = set(map(tuple, list(np.unique(arr_segmap.reshape(-1, arr_segmap.shape[2]), axis=0)))) \
            #     .difference({CONTEXT_BACKGROUND_COLOR})
            raise NotImplementedError("This part is not handled yet")
        label = np.zeros(arr_segmap.shape[:2], dtype=np.uint8)
        for color in unique_colors:
            # 1. Find all position in the png version that match the specific
            mask = (arr_segmap == color).all(axis=-1)
            # 2. Fill the corresponding position with the corresponding class index
            label[mask] = self.color_label_idx_mapping.get(color, BACKGROUND_LABEL)  # 1

            # This line is used to test the label is correct
            # label[mask] = 255.

        # 3. Convert from np array to torch tensor
        label = torch.from_numpy(label).long()  # long is int64
        return label
