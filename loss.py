import tensorflow as tf


class Loss:
    def __init__(self, loss_name):
        self.loss_name = loss_name

    def __call__(self, y_true, y_pred):
        # y_true.shape = y_pred.shape = [batch_size, ]
        if self.loss_name == "mean_squared":
            diff = y_true - y_pred
            square = tf.square(diff)
            mean = tf.reduce_mean(square, axis=0)
            return mean

        elif self.loss_name == "cross_entropy":
            ce = - (tf.multiply(y_true, tf.math.log(y_pred)) + tf.multiply(1-y_true, tf.math.log(1-y_pred)))
            mean = tf.reduce_mean(ce, axis=0)
            return mean
