#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Jun 15 19:02:54 2022

@author: Nacriema

Refs:

Test script - Sanity check the code
"""

# TESTING CODE for unet_plus_plus script

# from unet_plus_plus import NestedUNet
# import torch
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# net = NestedUNet(num_classes=1, deep_supervision=True).to(device)  # Num classes is 1 cus we can use BCELoss()
#
# # NCHW
# inp = torch.randn((1, 3, 720, 1080), requires_grad=True).to(device)
# output = net(inp)
# print("=======")


# TESTING CODE for resnet script
# import torch
#
# from utils.path import PROJECT_PATH
# import os
#
# from models.resnet import get_resnet_model
# from torchvision import transforms
# from PIL import Image

# os.environ['TORCH_HOME'] = str(PROJECT_PATH)

# Try to pull the pretrained model ResNet-34, not resolved, try another way
# Ok fixed, this due to the urllib
# Ok ResNet part is done !

# net = get_resnet_model('resnet34')(pretrained=True, progress=False)

# Check if the prediction is correct to confirm that the model is loaded success
# Ok PRETRAINED WEIGHT IS LOADED (both progress=True or False, the weight is loaded, but False is slightly faster)

# Transformation block
# preprocess = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     )
# ])

# img = Image.open('xxx.jpg')
#
# img_t = preprocess(img)
# batch_t = torch.unsqueeze(img_t, 0)
# net.eval()
# out = net(batch_t)
#
# _, index = torch.max(out, 1)
# percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
#
# print(index[0])
# print(percentage[index[0]])


# FIXING CERTIFICATION
# import certifi
# import requests
#
# try:
#     print("Checking connection to Github...")
#     test = requests.get("https://api.github.com")
#     print("Connection to Git hub OK.")
# except requests.exceptions.SSLError as err:
#     print("SSL Error. Adding custom certs to Certifi store...")

# import urllib.request
# import certifi
#
# This is a fixing Solution !!! This will help me when CODING
# urllib.request.urlopen("https://google.com/", cafile=certifi.where())
