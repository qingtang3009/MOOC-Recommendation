# -*- coding:utf-8 -*-
"""
Function: MLP model.
Input: Non.
Output: Non.
Author: Qing TANG
"""
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.network = nn.Sequential(
            # 128+(128+8)*706=96144
            nn.Linear(96144, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 706)
        )

    def forward(self, x):
        y_ = torch.sigmoid(self.network(x))
        # y_ = self.network(x)

        return y_
