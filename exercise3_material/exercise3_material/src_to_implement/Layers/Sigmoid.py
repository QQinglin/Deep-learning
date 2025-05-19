import numpy as np
from .Base import BaseLayer

class Sigmoid(BaseLayer):
    def __init__(self):
        super().__init__()
        self.output = None  # To store the output of the sigmoid function

    def forward(self, input_tensor):
        # Compute the sigmoid function: 1 / (1 + e^(-x))
        self.output = 1 / (1 + np.exp(-input_tensor))
        return self.output

    def backward(self, error_tensor):
        # Compute the gradient: error_tensor * sigmoid(x) * (1 - sigmoid(x))
        input_gradient = error_tensor * self.output * (1 - self.output)
        return input_gradient