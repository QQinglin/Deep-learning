from copy import deepcopy
import numpy as np
from exercise3_material.src_to_implement.Layers.Base import BaseLayer
from exercise3_material.src_to_implement.Layers.FullyConnected import FullyConnected
from exercise3_material.src_to_implement.Layers.Sigmoid import Sigmoid
from exercise3_material.src_to_implement.Layers.TanH import TanH

class RNN(BaseLayer):
    def __init__(self,input_size, hidden_size, output_size, memorize = False):
        super().__init__()

        self.trainable = True
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.memorize = memorize
        self.last_hidden = np.zeros((1, self.hidden_size))

        self.tanh = TanH()
        self.fc_hidden = FullyConnected(self.input_size + self.hidden_size, self.hidden_size)
        self.sigmoid = Sigmoid()
        self.fc_output = FullyConnected(self.hidden_size,self.output_size)

        self.gradient_weights = None

    @property
    def optimizer(self):
        return self.fc_hidden.optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self.fc_hidden.optimizer = optimizer
        self.fc_output.optimizer = deepcopy(optimizer)

    @property
    def weights(self):
        return self.fc_hidden.weights

    @weights.setter
    def weights(self, weights):
        self.fc_hidden.weights = weights

    def initialize(self, weights_initializer, bias_initializer):

        self.fc_hidden.initialize(weights_initializer, bias_initializer)
        self.fc_output.initialize(weights_initializer, bias_initializer)

    def initialize(self, weights_initializer, bias_initializer):

        self.fc_hidden.initialize(weights_initializer,bias_initializer)
        self.fc_output.initialize(weights_initializer,bias_initializer)

    def forward(self,input_tensor):
        self.x_tildes = []
        self.h_states = []
        self.fc_u = []
        self.fc_o = []
        self.y_states = np.zeros((input_tensor.shape[0], self.output_size))

        time_steps, input_dim = input_tensor.shape
        if not self.memorize:
            self.last_hidden = np.zeros((1,self.hidden_size))

        for t in range(time_steps):
            xt = input_tensor[t]
            x_t = np.expand_dims(xt, axis=0)
            x_tilde = np.hstack((x_t, self.last_hidden))
            self.x_tildes.append(x_tilde)

            u = self.fc_hidden.forward(self.x_tildes[t])
            self.fc_u.append(u)

            hidden_state = self.tanh.forward(u)
            self.h_states.append(hidden_state)

            o = self.fc_output.forward(hidden_state)
            self.fc_o.append(o)

            self.y_states[t] = self.sigmoid.forward(o)

            self.last_hidden = hidden_state
            output_tensor = self.y_states
        return output_tensor

    def backward(self,error_tensor):
        self.gradient_weights = 0
        accumulated = 0
        time_steps, output_dim = error_tensor.shape
        error_tensor_to_previous = np.zeros((error_tensor.shape[0], self.input_size))

        for t in reversed(range(time_steps)):
            dy_t = error_tensor[t]

            self.sigmoid.forward(self.fc_o[t])
            do_t = self.sigmoid.backward(dy_t)

            self.fc_output.input_tensor = np.hstack((self.h_states[t], np.ones((1, 1))))
            dh_t = self.fc_output.backward(do_t) + accumulated

            self.tanh.forward(self.fc_u[t])
            du_t = self.tanh.backward(dh_t)

            self.fc_hidden.input_tensor = np.hstack((self.x_tildes[t], np.ones((1, 1))))
            grad_x_tilde = self.fc_hidden.backward(du_t)

            self.gradient_weights += self.fc_hidden.gradient_weights

            error_tensor_to_previous[t] = grad_x_tilde[0,:self.input_size]
            accumulated = grad_x_tilde[:, self.input_size:]

        return error_tensor_to_previous