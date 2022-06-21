#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Jun 21 14:55:30 2022

@author: Nacriema

Refs:

"""
import torch
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

# Test with loss writer during training
x = torch.arange(-5, 5, 0.1).view(-1, 1)
y = -5 * x + 0.1 * torch.randn(x.size())

model = torch.nn.Linear(1, 1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


def train_model(iter):
    for epoch in range(iter):
        y1 = model(x)
        loss = criterion(y1, y)
        writer.add_scalar("Loss/train", loss, epoch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


train_model(10)  # Train for 10 epochs
writer.flush()
print(model.weight)
print(model.bias)
