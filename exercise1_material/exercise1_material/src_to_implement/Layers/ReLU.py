import numpy as np
import torch
from .Base import BaseLayer

class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()
        self.trainable = False

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        result = np.maximum(0, input_tensor)
        return result

    def backward(self, error_tensor):
        mask = np.where(self.input_tensor > 0, 1.0, 0.0)
        result = mask * error_tensor
        return result