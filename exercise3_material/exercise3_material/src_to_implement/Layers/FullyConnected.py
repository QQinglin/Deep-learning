
import numpy as np
from .Base import BaseLayer

class FullyConnected(BaseLayer) :
    def __init__(self, input_size, output_size):
       super().__init__()
       self.trainable = True
       self.weights = np.random.uniform(low = 0, high = 1,size = (input_size + 1,output_size))
       self.input_size = input_size
       self.output_size = output_size
       self.gradient_weights = None
       self.optimizer = None
       self.input_tensor = None
       self.gradient_weights = None


    def forward(self,input_tensor):
        bias = np.ones((input_tensor.shape[0],1))
        input_tensor_bias = np.hstack((input_tensor,bias))
        self.input_tensor = input_tensor_bias.copy()
        output = input_tensor_bias @ self.weights #+ self.weights_bias
        return output

    def backward(self,error_tensor):
        gradient_input = error_tensor @ self.weights.T[:,:-1]
        self.gradient_weights = self.input_tensor.T @ error_tensor
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights,self.gradient_weights)
        return gradient_input

    def initialize(self, weights_initializer, bias_initializer):
        weights = weights_initializer.initialize((self.input_size,self.output_size),self.input_size,self.output_size)
        bias = bias_initializer.initialize((1,self.output_size),self.input_size,self.output_size)

        self.weights = np.vstack((weights,bias))

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer