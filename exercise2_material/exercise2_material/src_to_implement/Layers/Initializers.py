import numpy as np
import math

class Constant:
    def __init__(self,value=0.1):
        self.value = value

    def initialize(self,weights_shape,fan_in,fan_out):
        return np.full(weights_shape,self.value)

class UniformRandom:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.uniform(low=0,high=1,size=weights_shape)

class Xavier:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2/(fan_in + fan_out))
        return np.random.normal(loc=0.0,scale=sigma,size=weights_shape)

class He:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = math.sqrt(2/fan_in)
        return np.random.normal(loc=0.0, scale=sigma, size=weights_shape)



