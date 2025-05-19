import numpy as np

from .Base import BaseLayer

class Pooling(BaseLayer):
    def __init__(self,stride_shape,pooling_shape):
        self.output_tensor = None
        self.input_tensor = None
        self.max_indices = None
        self.output_width = None
        self.output_height = None
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.trainable = None


    def forward(self,input_tensor):
        self.input_tensor = input_tensor

        batch_size, channels, height, width = input_tensor.shape
        pool_height, pool_width = self.pooling_shape
        stride_height, stride_width = self.stride_shape

        self.output_height = (height - pool_height) // stride_height + 1
        self.output_width = (width - pool_width) // stride_width + 1

        output_tensor = np.zeros((batch_size,channels,self.output_height,self.output_width))
        self.max_indices = np.zeros((batch_size, channels, self.output_height, self.output_width, 2), dtype=int)

        for b in range(batch_size):
            for c in range(channels):
                for i in range(self.output_height):
                    for j in range(self.output_width):
                        hei_begin, hei_end = i * stride_height, i * stride_height + pool_height
                        width_begin, width_end = j * stride_width, j * stride_width + pool_width

                        window = input_tensor[b,c,hei_begin:hei_end,width_begin:width_end]
                        output_tensor[b,c,i,j] = np.max(window)

                        max_indices_h, max_indices_w = np.unravel_index(np.argmax(window), window.shape)

                        self.max_indices[b,c,i,j,0] = hei_begin + max_indices_h
                        self.max_indices[b,c,i,j,1] = width_begin + max_indices_w
        self.input_tensor = input_tensor
        self.output_tensor = output_tensor
        return output_tensor

    def backward(self,error_tensor):
        batch_size, channels,height, width  = error_tensor.shape
        input_gradient = np.zeros_like(self.input_tensor,dtype=np.float32)
        for b in range(batch_size):
            for c in range(channels):
                for i in range(self.output_height):
                    for j in range(self.output_width):
                        max_h, max_w = self.max_indices[b, c, i, j]
                        input_gradient[b,c,max_h,max_w] += error_tensor[b,c,i,j]

        return input_gradient
