import tensorflow as tf
from . import layer

class Model():
    def __init__(self, layer_sizes, input_size):
        input_sizes = layer_sizes.push(input_size)[:-1]
        layers = []
        for size, input_size in zip(layer_sizes, input_sizes):
            layers.append(layer.Layer(size, input_size))

    def __call__(self):
        pass