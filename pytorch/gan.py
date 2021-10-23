import torch.nn as nn
import torch as torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(100, 1024, bias=False)
        self.bn1 = nn.BatchNorm1d(1024)
        self.act1 = nn.LeakyReLU(inplace=True)
        self.l2 = nn.Linear(1024,784)
        self.act2 = nn.Sigmoid()
    def forward(self, z):
        y = self.l1(z)
        y = self.bn1(y)
        y = self.act1(y)
        y = self.l2(y)
        y = self.act2(y)
        return y

class Disc(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(784, 512)
        self.act1 = nn.LeakyReLU(inplace=True)
        self.l2 = nn.Linear(512,1)
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        y = self.l1(x)
        y = self.act1(y)
        y = self.l2(y)
        y = self.act2(y)
        return y
