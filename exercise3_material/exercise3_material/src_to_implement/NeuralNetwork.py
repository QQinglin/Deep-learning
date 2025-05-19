import copy
import pickle
from copy import deepcopy

class NeuralNetwork:
    def __init__(self,optimizer,weights_initializer,bias_initializer):

        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.loss_layer = None
        self.data_layer = None

        self.weight_initializer = weights_initializer
        self.bias_initializer = bias_initializer


        self._phase = False


    def forward(self):
        input_tensor,label_tensor = self.data_layer.next()
        self.input_tensor = input_tensor
        self.label_tensor = label_tensor

        regularization_loss = 0
        data_tensor = input_tensor.copy()

        for i,layer in enumerate(self.layers):
            data_tensor = layer.forward(data_tensor)

            # sum up the regularization loss in each layer
            # and layer.optimizer is not None and layer.optimizer.regularizer is not None
            if layer.trainable and layer.optimizer is not None and layer.optimizer.regularizer is not None:
                regularization_loss += layer.optimizer.regularizer.norm(layer.weights)
        loss = self.loss_layer.forward(data_tensor,label_tensor) + regularization_loss
        return loss

    def backward(self):

        error_tensor = self.loss_layer.backward(self.label_tensor)

        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)


    def append_layer(self,layer):

        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
            layer.initialize(self.weight_initializer,self.bias_initializer)

        self.layers.append(layer)


    def train(self,iterations):
        self._phase = False

        if self.layers is not None:
            for layer in self.layers:
                layer.testing_phase = self._phase

        for _ in range(iterations):
            loss = self.forward()
            self.backward()
            self.loss.append(loss)


    def test(self,input_tensor):
        self._phase = True

        if self.layers is not None:
            for layer in self.layers:
                layer.testing_phase = self._phase

        data_tensor = input_tensor.copy()
        for layer in self.layers:
            data_tensor = layer.forward(data_tensor)

        return data_tensor