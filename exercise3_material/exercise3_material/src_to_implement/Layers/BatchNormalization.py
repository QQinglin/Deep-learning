from copy import deepcopy
from .Base import BaseLayer
import numpy as np
from exercise3_material.src_to_implement.Layers import Helpers
class BatchNormalization(BaseLayer):

    def __init__(self, channels):
        super().__init__()
        self.bias = None
        self.weights = None
        self.gradient_weights = 0
        self.gradient_bias = 0
        self.trainable = True
        self.channels = channels
        self.epsilon = 1e-10
        self.alpha = 0.8

        self.initialize(None,None)

        self.mu_tilde = None
        self.sigma_square_tilde = None
        self.input_tensor = None

        self._optimizer = None
        self._optimizer1 = None

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self,optimizer):
        self._optimizer = optimizer
        self._optimizer1 = deepcopy(optimizer)

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = np.ones(self.channels)
        self.bias =  np.zeros(self.channels)

    def _forward(self,input_tensor):
        if not self.testing_phase:
            if self.sigma_square_tilde is None and self.mu_tilde is None:
                self.sigma_square_tilde = np.var(input_tensor, axis=0)
                self.mu_tilde = np.mean(input_tensor, axis=0)
            else:
                self.sigma_square_tilde = self.alpha * self.sigma_square_tilde + (1 - self.alpha) * np.var(input_tensor, axis=0)
                self.mu_tilde = self.alpha * self.mu_tilde + (1 - self.alpha) * np.mean(input_tensor, axis=0)
        if self.testing_phase:
            x_tilde = (input_tensor - self.mu_tilde) / np.sqrt(self.sigma_square_tilde + self.epsilon)
            y_hat = self.weights * x_tilde + self.bias
            return y_hat
        else:
            x_tilde = (input_tensor - np.mean(input_tensor, axis=0)) / np.sqrt(np.var(input_tensor, axis=0) + self.epsilon)
            y_hat = self.weights * x_tilde + self.bias
            return y_hat

    def forward(self,input_tensor):
        self.input_tensor = input_tensor.copy()

        # one image with only one channel (height, width)
        if len(self.input_tensor.shape) == 2:
            return self._forward(input_tensor)
        else:
            # (batch_size, channels, height, width)
            # (batch_size * height * width, channels)
            input_reshaped = self.reformat(input_tensor)
            output_tensor = self._forward(input_reshaped)
            return self.reformat(output_tensor)

    def backward(self,error_tensor):
        if len(self.input_tensor.shape) > 2:
            error_reshaped = self.reformat(error_tensor)
            input_reshaped = self.reformat(self.input_tensor)
        else:
            error_reshaped = error_tensor.copy()
            input_reshaped = self.input_tensor.copy()

        self.gradient_input = Helpers.compute_bn_gradients(error_reshaped, input_reshaped, self.weights, self.mu_tilde, self.sigma_square_tilde)

        x_b_tilde = (input_reshaped - np.mean(input_reshaped,axis=0)) / (np.sqrt(np.var(input_reshaped,axis=0) + self.epsilon))

        self.gradient_weights = np.sum(error_reshaped * x_b_tilde,axis=0)
        self.gradient_bias = np.sum(error_reshaped,axis=0)

        if self._optimizer is not None and self._optimizer1 is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
            self.bias = self._optimizer1.calculate_update(self.bias, self.gradient_bias)

        gradient_input = self.gradient_input
        if len(self.input_tensor.shape) > 2:
            return self.reformat(gradient_input)
        else:
            return gradient_input

    def reformat(self,tensor):
        if len(tensor.shape) == 4:
            self.batch_size, self.channels, self.height, self.width = tensor.shape
            # (batch_size, channels, height * width)
            output_tensor = tensor.reshape(self.batch_size, self.channels, -1)
            # (batch_size, height * width, channels)
            output_tensor = np.transpose(output_tensor, (0, 2, 1))
            # (batch_size * height * width, channels)
            output_tensor = output_tensor.reshape(-1, self.channels)
            return output_tensor

        elif len(tensor.shape) == 2:
            total_elements, channels = tensor.shape
            if self.batch_size is None or self.height is None or self.width is None:
                raise ValueError("Shape attributes (batch_size, height, width) not set. Please run forward pass first.")

            output_tensor = tensor.reshape(self.batch_size, self.height * self.width,
                                           channels)  # (batch_size, height * width, channels)
            output_tensor = np.transpose(output_tensor, (0, 2, 1))  # (batch_size, channels, height * width)
            output_tensor = output_tensor.reshape(self.batch_size, channels, self.height,
                                                  self.width)  # (batch_size, channels, height, width)
            return output_tensor

        else:
            raise ValueError("Tensor must be 2D or 4D")

