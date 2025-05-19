import os
import copy
import math
import numpy as np
from scipy.signal import convolve, correlate
from .Base import BaseLayer
from exercise2_material.src_to_implement.Optimization.Optimizers import Sgd


class Conv(BaseLayer):
    def __init__(self,stride_shape,convolution_shape,num_kernels):
        super().__init__()

        self.input_tensor = None
        self._optimizer = None
        self._optimizer2 = None

        self.trainable = True

        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels

        self.weights = np.random.uniform(0, 1, (num_kernels,) + self.convolution_shape)
        self.bias = np.random.uniform(0, 1, num_kernels)

        self.gradient_weights = 0
        self.gradient_bias = 0



    def initialize(self, weights_initializer, bias_initializer):

        self.weights = weights_initializer.initialize((self.num_kernels,) + self.convolution_shape, np.prod(self.convolution_shape),
                                                      np.prod((self.num_kernels,) + self.convolution_shape[1:]))
        self.bias = bias_initializer.initialize(self.num_kernels, np.prod(self.convolution_shape),
                                                np.prod((self.num_kernels,) + self.convolution_shape[1:]))

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self._optimizer2 = copy.deepcopy(self._optimizer)


    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        batch_size, channels, *spatial_dims = input_tensor.shape

        batch_layers = []
        for b in range(batch_size):
            layers = []
            for k in range(self.num_kernels):
                single_layer = 0
                for c in range(channels):
                    single_layer += correlate(input_tensor[b,c,:], self.weights[k,c], mode='same')

                stride_height = self.stride_shape[0]
                if len(spatial_dims) == 2:
                    stride_width = self.stride_shape[1]
                    single_layer = single_layer[0::stride_height, 0::stride_width] + self.bias[k]
                else:
                    single_layer = single_layer[0::stride_height] + self.bias[k]
                layers.append(single_layer)
            batch_layers.append(layers)
            output_tensor = np.array(batch_layers)
        return output_tensor

    def backward(self, error_tensor):
        _, _, *spatial_dims = error_tensor.shape
        batch_size, input_channels, *input_spatial_dims = self.input_tensor.shape

        upsampled_error_tensors = np.zeros((batch_size, self.num_kernels,*input_spatial_dims))

        if len(self.stride_shape) == 2:
            upsampled_error_tensors[:,:,::self.stride_shape[0],::self.stride_shape[1]] = error_tensor
            pad = self.padding2D
            unpad = self.unpad2D
        else:
            upsampled_error_tensors[:,:,::self.stride_shape[0]] = error_tensor
            pad = self.padding1D
            unpad = self.unpad1D

        input_gradient = np.zeros((batch_size, *self.input_tensor[0].shape))
        kernel_gradient = np.zeros((batch_size,self.num_kernels,*self.convolution_shape))
        gradient_bias = np.zeros((batch_size,self.num_kernels))

        for b, error_tensors in enumerate(upsampled_error_tensors):
            sample = self.input_tensor[b]
            for k in range(self.num_kernels):
                for channel, image in enumerate(sample):
                    padded_image = pad(image, *self.convolution_shape[1:])
                    kernel_gradient[b,k,channel] = correlate(padded_image,error_tensors[k], mode='valid')

                    t = convolve(error_tensors[k],self.weights[k,channel], mode='full')
                    input_gradient[b,channel] += unpad(t, *self.convolution_shape[1:])

                gradient_bias[b,k] = np.sum(error_tensors[k])

        self.gradient_weights = np.sum(kernel_gradient, axis=0)
        self.gradient_bias = np.sum(gradient_bias, axis=0)

        if self._optimizer is not None and self._optimizer2 is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)
            self.bias = self._optimizer2.calculate_update(self.bias, self.gradient_bias)

        return input_gradient

    def unpad2D(self,padded_sample, kernel_height, kernel_width):
        width_up = math.ceil((kernel_width - 1) / 2)
        width_down = math.floor((kernel_width - 1) / 2)

        height_up = math.ceil((kernel_height - 1) / 2)
        height_down = math.floor((kernel_height - 1) / 2)
        return  padded_sample[height_up:padded_sample.shape[0] - height_down,width_up:padded_sample.shape[1] - width_down]

    def padding2D(self,sample, kernel_height, kernel_width):
        width_up = math.ceil((kernel_width - 1) / 2)
        width_down = math.floor((kernel_width - 1) / 2)

        height_up = math.ceil((kernel_height - 1) / 2)
        height_down = math.floor((kernel_height - 1) / 2)
        padded_sample = np.zeros(np.array(sample.shape) + (height_up + height_down, width_up + width_down))

        padded_sample[height_up:height_up + sample.shape[0], width_up:width_up + sample.shape[1]] = sample
        return padded_sample

    def padding1D(self,sample, kernel_width):
        width_up = math.ceil((kernel_width - 1) / 2)
        width_down = math.floor((kernel_width - 1) / 2)

        padded_sample = np.zeros(np.array(sample.shape[0] + width_up + width_down))
        padded_sample = padded_sample[width_up:width_up + sample.shape[0]]
        return padded_sample

    def unpad1D(self,padded_sample, kernel_width):
        width_up = math.ceil((kernel_width - 1) / 2)
        width_down = math.floor((kernel_width - 1) / 2)

        return padded_sample[width_up:padded_sample.shape[0] - width_down]


