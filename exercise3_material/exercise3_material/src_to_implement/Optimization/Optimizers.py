import numpy as np
epsilon = 1e-8

class Optimizer:
    def __init__(self):
        self.regularizer = None

    def add_regularizer(self,regularizer):
        self.regularizer = regularizer


class Sgd(Optimizer):
    def __init__(self,learning_rate):
        super().__init__()
        self.learning_rate = learning_rate

    def calculate_update(self,weight_tensor, gradient_tensor):
        update_weights = weight_tensor - self.learning_rate * gradient_tensor
        if self.regularizer is not None:
            gradient_regularizer = self.regularizer.calculate_gradient(weight_tensor)
        else:
            gradient_regularizer = 0
        return weight_tensor - self.learning_rate * gradient_regularizer - self.learning_rate * gradient_tensor


class SgdWithMomentum(Optimizer):
    def __init__(self,learning_rate,momentum_rate):
        super().__init__()

        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v = 0

    def calculate_update(self,weight_tensor,gradient_tensor):
        if self.regularizer is not None:
            gradient_regularizer = self.regularizer.calculate_gradient(weight_tensor)
        else:
            gradient_regularizer = 0

        self.v = self.momentum_rate * self.v - self.learning_rate * gradient_tensor
        return weight_tensor + self.v - self.learning_rate * gradient_regularizer

class Adam(Optimizer):
    def __init__(self,learning_rate,mu,rho):
        super().__init__()

        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.epsilon = 1e-8
        self.v = None
        self.r = None
        self.k = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.v is None:
            self.v = np.zeros_like(weight_tensor)
        if self.r is None:
            self.r = np.zeros_like(weight_tensor)

        self. k = self.k + 1
        g = gradient_tensor
        self.v = self.mu * self.v + (1 - self.mu) * g
        self.r = self.rho * self.r + (1 - self.rho) * (g * g)

        # Bias correction:
        v_hat = self.v / (1 - self.mu ** self.k + self.epsilon)
        r_hat = self.r / (1 - self.rho ** self.k + self.epsilon)

        update = self.learning_rate * v_hat / (np.sqrt(r_hat) + self.epsilon)
        if self.regularizer is not None:
            gradient_regularizer = self.regularizer.calculate_gradient(weight_tensor)
        else:
            gradient_regularizer = 0

        weight_tensor = weight_tensor - update - self.learning_rate * gradient_regularizer
        return  weight_tensor

