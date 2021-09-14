import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
from tqdm import tqdm


class Generator(tfk.Model):
    def __init__(self):
        super().__init__()
        self.l1 = tfkl.Dense(1024, activation="leaky_relu")
        self.bn1 = tfkl.BatchNormalization()
        self.l2 = tfkl.Dense(784, activation="sigmoid")
    def call(self, x):
        y = self.l1(x)
        y = self.bn1(y)
        y = self.l2(y)
        return y

class Disc(tfk.Model):
    def __init__(self):
        super().__init__()
        self.l1 = tfkl.Dense(512, activation="leaky_relu")
        self.l2 = tfkl.Dense(1, activation="sigmoid")
    def call(self, x):
        y = self.l1(x)
        y = self.l2(y)
        return y
