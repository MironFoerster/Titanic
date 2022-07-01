import tensorflow as tf
import layer


class Model:
    def __init__(self, layer_sizes, input_size, activation):
        layer_sizes.insert(0, input_size)
        input_sizes = layer_sizes[:-1]
        self.layers = []
        self.vars = []

        for idx, (size, input_size) in enumerate(zip(layer_sizes, input_sizes)):
            if idx == len(layer_sizes)-1:
                activation = tf.sigmoid
            self.layers.append(layer.Layer(size, input_size, activation))
            self.vars += self.layers[-1].vars
        print(self.vars)

    def __call__(self, features):
        # features.shape: [num_features, batch_size]
        for l in self.layers:
            features = l(features)
            # features.shape: [layer_size, batch_size]
        return features
