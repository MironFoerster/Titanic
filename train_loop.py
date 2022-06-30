import tensorflow as tf
from . import model

titanic_model = model.Model(layer_sizes=[128, 64, 32, 8, 2])


with tf.GradientTape as tape:
    network_output =


gradients = tape.gradient(network_output, trainable_variables)