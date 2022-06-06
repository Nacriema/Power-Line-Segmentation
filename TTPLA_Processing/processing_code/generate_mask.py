#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Jun 06 17:09:53 2022

@author: Nacriema

Generate mask image from and json annotated file

Refs:
https://stackoverflow.com/questions/50805634/how-to-create-mask-images-from-coco-dataset
Hmm that's not good, we need a custom created dataset for this !

Each file json in format:

{
    "version": "4.2.7",
    "flags": {},
    "shapes": [
        {
            "label": "cable",
            "line_color": null,
            "fill_color": null,
            "points": [
                [
                  2731.944444444445,
                  1260.0
                ]
            ],
            "shape_type": "polygon",
            "flags": {}
        },
        {
            "label": "cable",
            "line_color": null,
            "fill_color": null,
            "points": [
                [
                  2731.944444444445,
                  1260.0
                ]
            ],
            "shape_type": "polygon",
            "flags": {}
        },
    ],
    "lineColor": [
        0,
        255,
        0,
        128
    ],
    "fillColor": [
        255,
        0,
        0,
        128
    ],
    "imagePath": "xxx.jpg",
    "imageData": base64 format,
    "imageHeight": 2160,
    "imageWidth": 3840
}
"""

# Create another COCO class for this type of annotation
# have method to create the groundTruth image

import json
import time
from PIL import Image, ImageDraw
import numpy as np


class TTPLAObject:
    def __init__(self, label, line_color, fill_color, points, shape_type, flags):
        self.label = label
        self.line_color = line_color
        self.fill_color = fill_color
        self.points = points
        self.shape_type = shape_type
        self.flags = flags

    def __str__(self):
        info = f'Label: {self.label}, \n' \
               f'Line color: {self.line_color}, \n' \
               f'Fill color: {self.fill_color}, \n' \
               f'Points: {self.points}, \n' \
               f'Shape type: {self.shape_type}, \n' \
               f'Flags: {self.flags} \n'
        return info


class TTPLA:
    def __init__(self, image_folder, annotation_file=None):
        """
        Construct COCO helper class for TTPLA dataset
        :param annotation_file (str): location of annotation file
        :return:
        """
        # load dataset
        self.dataset = dict()
        self.image_folder = image_folder
        if annotation_file is not None:
            print('Loading annotations into memory...')
            tic = time.time()
            dataset = json.load(open(annotation_file, 'r'))
            assert type(dataset) == dict, 'Annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time() - tic))
            self.dataset = dataset
            self._createData()

    def _createData(self):
        # create data
        print("Creating data...")
        lineColor, fillColor, imagePath, imageHeight, imageWidth = None, None, None, None, None
        objects = []  # Annotations
        if 'lineColor' in self.dataset:
            lineColor = self.dataset['lineColor']
        if 'fillColor' in self.dataset:
            fillColor = self.dataset['fillColor']
        if 'imagePath' in self.dataset:
            imagePath = self.dataset['imagePath']
        if 'imageHeight' in self.dataset:
            imageHeight = self.dataset['imageHeight']
        if 'imageWidth' in self.dataset:
            imageWidth = self.dataset['imageWidth']
        if 'shapes' in self.dataset:
            for obj in self.dataset['shapes']:
                objects.append(TTPLAObject(obj['label'],
                                           obj['line_color'],
                                           obj['fill_color'],
                                           self._nestedList2Tuple(obj['points']),
                                           obj['shape_type'],
                                           obj['flags']))
        # Create class member
        self.lineColor = lineColor
        self.fillColor = fillColor
        self.imagePath = imagePath
        self.imageHeight = imageHeight
        self.imageWidth = imageWidth
        self.objects = objects

    def _nestedList2Tuple(self, lst):
        return [tuple(_) for _ in lst]

    def _markup(self, image):
        """
        Draw the segments into an image
        :param image:
        :return:
        """
        draw = ImageDraw.Draw(image, 'RGB')
        for obj in self.objects:
            if obj.shape_type == 'polygon' and obj.label == 'cable':
                draw.polygon(obj.points, fill=(255, 255, 255))
        del draw
        return image

    def generateLabel(self, output_folder):
        """
        :param output_folder:
        :return:
        """
        img_ = Image.new('RGB', (self.imageWidth, self.imageHeight), color=(0, 0, 0))
        labelImg = self._markup(img_)
        labelImg.save(f'./{self.imagePath.split(".")[0]}_Label.png')

    def generateExample(self, output_folder):
        """
        Generate so-called an example for the current image
        :param output_folder:
        :return:
        """
        imagePath = self.image_folder + '/' + self.imagePath
        with Image.open(imagePath) as currentImage:
            draw = ImageDraw.Draw(currentImage, 'RGBA')
            for obj in self.objects:
                if obj.shape_type == 'polygon':
                    random_color = tuple(np.append(np.random.choice(range(256), size=3), 125))
                    draw.polygon(obj.points, fill=random_color)
            del draw
            currentImage.save(f'./{self.imagePath.split(".")[0]}_Sample.png')


json_file = '../data_sample/04_2220.json'
data = TTPLA('../data_sample', json_file)
# print(data.dataset)
print(data.objects[0])
data.generateLabel(None)
data.generateExample(None)
