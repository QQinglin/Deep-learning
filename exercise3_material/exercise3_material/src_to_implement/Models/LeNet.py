from exercise3_material.src_to_implement.Layers.Conv import Conv
from exercise3_material.src_to_implement.Layers.Flatten import Flatten
from exercise3_material.src_to_implement.Layers.FullyConnected import FullyConnected
from exercise3_material.src_to_implement.Layers.Initializers import UniformRandom
from exercise3_material.src_to_implement.Layers.Pooling import Pooling
from exercise3_material.src_to_implement.Layers.Sigmoid import Sigmoid
from exercise3_material.src_to_implement.NeuralNetwork import NeuralNetwork
from exercise3_material.src_to_implement.Optimization.Constraints import L2_Regularizer
from exercise3_material.src_to_implement.Optimization.Loss import CrossEntropyLoss
from exercise3_material.src_to_implement.Optimization.Optimizers import Adam


def build():

    regulizer = L2_Regularizer(4e-4)
    optimizer = Adam(5e-4,0.9,0.999)
    optimizer.add_regularizer(regulizer)
    weights_initializer = UniformRandom()
    bias_initializer = UniformRandom()
    net = NeuralNetwork(optimizer,weights_initializer,bias_initializer)



    net.append_layer(Conv(stride_shape=(1,1),convolution_shape=(1,5,5),num_kernels=6))
    net.append_layer(Sigmoid())
    net.append_layer(Pooling(stride_shape=(2,2),pooling_shape=(2,2)))
    net.append_layer(Conv(stride_shape=(1,1),convolution_shape=(6,5,5),num_kernels=16))
    net.append_layer(Sigmoid())
    net.append_layer(Pooling((2,2),pooling_shape=(2,2)))
    net.append_layer(Flatten())
    net.append_layer(FullyConnected(784,84))
    net.append_layer(Sigmoid())
    net.append_layer(FullyConnected(84,10))
    net.append_layer(Sigmoid())
    net.append_layer(FullyConnected(10,10))

    net.loss_layer = CrossEntropyLoss()



    return net