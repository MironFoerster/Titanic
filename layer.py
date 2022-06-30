import tensorflow as tf

class Layer(tf.keras.layers.Layer):
    def __init__(self, size, input_size):
        self.W_initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
        self.b_initializer = tf.keras.initializers.Zeros()

        self.W = self.W_initializer(shape=(size, input_size))
        self.b = self.b_initializer(shape=(input_size))

    def call(self, inputs):
