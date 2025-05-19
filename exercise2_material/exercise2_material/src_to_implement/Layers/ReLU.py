import numpy as np
import torch

class ReLU:
    def __init__(self):
        self.trainable = False
        pass

    def forward(self, input_tensor):
        self.input_tensor = input_tensor

        # Apply ReLU: max(0, x)
        if isinstance(input_tensor, torch.Tensor):
            # Using PyTorch
            result = torch.maximum(torch.tensor(0, dtype=input_tensor.dtype), input_tensor)
        else:
            # Using NumPy as fallback
            result = np.maximum(0, input_tensor)

        return result

    def backward(self, error_tensor):
        if isinstance(self.input_tensor, torch.Tensor):
            # Using PyTorch
            # Create a mask where input > 0
            mask = (self.input_tensor > 0).float()
            result = mask * error_tensor
        else:
            # Using NumPy as fallback
            mask = np.where(self.input_tensor > 0, 1.0, 0.0)
            result = mask * error_tensor

        return result