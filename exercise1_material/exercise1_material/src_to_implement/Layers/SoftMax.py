import numpy as np
import torch
from scipy.fftpack import shift


class SoftMax:
    def __init__(self):
        self.trainable = False
        self.output = None

    def forward(self,input_tensor):
        self.input_tensor = input_tensor
        shifted_input = input_tensor - np.max(input_tensor, axis=1, keepdims=True)
        exp_values = np.exp(shifted_input)
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return  self.output

    def backward(self,error_tensor):
        sum_over_class = np.sum(error_tensor * self.output, axis=1, keepdims=True)
        grad_input = self.output * (error_tensor - sum_over_class)
        return grad_input