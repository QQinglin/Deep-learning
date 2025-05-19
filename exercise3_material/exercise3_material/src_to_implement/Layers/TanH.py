import numpy as np
from .Base import BaseLayer

class TanH(BaseLayer):
    def __init__(self):
        super().__init__()
        self.cache = None

    def forward(self,input_tensor):
        output_tensor = np.tanh(input_tensor)
        self.cache = output_tensor
        return output_tensor

    def backward(self,error_tensor):
        tanh = self.cache
        input_gradient = error_tensor * (1 - tanh**2)
        return input_gradient
