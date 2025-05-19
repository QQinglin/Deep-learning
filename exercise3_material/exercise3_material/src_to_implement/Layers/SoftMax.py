import numpy as np
import torch
from scipy.fftpack import shift
import math

import numpy as np

from .Base import BaseLayer

class SoftMax(BaseLayer):
    def __init__(self):
        self.trainable = False
        self.output = None

    def forward(self,input_tensor):
        self.input_tensor = input_tensor
        if isinstance(input_tensor, torch.Tensor):
            max_values,_ = torch.max(input_tensor,dim=1,keepdim=True)
            shift_input,_ = input_tensor - max_values
            exp_values = torch.exp(shift_input)
            self.output = exp_values/torch.sum(exp_values,dim=1,keepdim=True)
        else:
            shifted_input = input_tensor - np.max(input_tensor, axis=1, keepdims=True)
            exp_values = np.exp(shifted_input)
            self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return  self.output

    def backward(self,error_tensor):
        if isinstance(error_tensor, torch.Tensor):
            sum_over_class = torch.sum(error_tensor * self.output, dim=1, keepdim=True)
            grad_input = self.output * (error_tensor - sum_over_class)
        else:
            sum_over_class = np.sum(error_tensor * self.output, axis=1, keepdims=True)
            grad_input = self.output * (error_tensor - sum_over_class)

        return grad_input