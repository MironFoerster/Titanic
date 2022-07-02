import tensorflow as tf
import tensorflow_datasets as tfds
import model
import loss
import data

learning_rate = 0.01
batch_size = 32

titanic_model = model.Model(layer_sizes=[128, 64, 32, 8, 2], input_size=11, activation=tf.tanh)
loss = loss.Loss("mean_squared")
titanic_dataset = tfds.load("titanic", split='train', shuffle_files=True, as_supervised=True)
features_removed_ds = titanic_dataset.map(data.remove_features)

preprocessed_ds = features_removed_ds.map(data.preprocess_features)

for features, y_true in preprocessed_ds.batch(batch_size):
    with tf.GradientTape() as tape:
        network_output = titanic_model(features)
        loss = loss(y_true, network_output)
        print(loss.numpy())

    gradients = tape.gradient(loss, titanic_model.vars)

    # update
    for var, grad in zip(titanic_model.vars, gradients):
        var.assign_sub(tf.multiply(learning_rate, grad))
