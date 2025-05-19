from .Base import BaseLayer
import numpy as np

class Dropout(BaseLayer):
    def __init__(self,probability):
        super().__init__()
        self.probability = probability


    def forward(self,input_tensor):
        if self.testing_phase is False:
            # make the position where number smaller than p equal to 1 to preserve
            # E[output] = p⋅x + (1−p)⋅0 = p⋅x; E[output] = p⋅x/p + (1−p)⋅0 = x because of expectation
            self.mask = (np.random.rand(*input_tensor.shape) < self.probability).astype(float)
            output_tensor = input_tensor * self.mask / self.probability
        else:
            output_tensor = input_tensor
        return output_tensor

    def backward(self,error_tensor):
        # forward : y_hat = x*mask/p so that derivative of backward is mask/p
        if self.testing_phase is False:
            error_tensor_prev = error_tensor * self.mask / self.probability
        else:
            error_tensor_prev = error_tensor / self.probability

        return error_tensor_prev