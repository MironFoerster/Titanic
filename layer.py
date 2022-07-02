import tensorflow as tf


class Layer(tf.keras.layers.Layer):
    def __init__(self, size, input_size, activation):
        self.activation = activation
        self.W_initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
        self.b_initializer = tf.keras.initializers.Zeros()

        self.W = self.W_initializer(shape=(size, input_size))
        self.b = self.b_initializer(shape=(size, 1))
        self.vars = [self.W, self.b]

    def __call__(self, inputs):
        # self.W.shape: [layer_size, input_size]
        # inputs.shape: [input_size, batch_size]
        return self.activation(tf.matmul(self.W, inputs) + self.b)  # shape: [layer_size, batch_size]
