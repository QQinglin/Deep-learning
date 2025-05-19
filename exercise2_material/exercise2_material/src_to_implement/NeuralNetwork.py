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

    def forward(self):
        self.input_tensor, self.label_tensor = self.data_layer.next()
        current_output = self.input_tensor.copy()
        for current_layer in self.layers:
            current_output = current_layer.forward(current_output)

        loss_output = self.loss_layer.forward(current_output,self.label_tensor)
        return  loss_output

    def backward(self):
        grad_input = self.loss_layer.backward(self.label_tensor)
        for current_layer in reversed(self.layers):
            grad_input = current_layer.backward(grad_input)

    def append_layer(self,layer):
        if layer.trainable is True:
                layer.optimizer = deepcopy(self.optimizer)
                layer.initialize(self.weight_initializer,self.bias_initializer)

        self.layers.append(layer)

    def train(self,iterations):
        for i in range(iterations):
            loss_iteration = self.forward()
            self.loss.append(loss_iteration)
            self.backward()


    def test(self,input_tensor):
        current_output = input_tensor.copy()
        for current_layer in self.layers:
            current_output = current_layer.forward(current_output)

        #prediction = self.loss_layer.forward(current_output, self.label_tensor,None)
        return current_output

