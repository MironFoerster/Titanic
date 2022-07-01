import tensorflow as tf
import tensorflow_datasets as tfds
import model
import loss


learning_rate = 0.01
batch_size = 32

titanic_model = model.Model(layer_sizes=[128, 64, 32, 8, 2], input_size=30, activation=tf.tanh)
loss = loss.Loss("mean_squared")
titanic_dataset = tfds.load("titanic", split='train', shuffle_files=True, as_supervised=True)
for i in titanic_dataset:
    print(i)
prepared_ds = titanic_dataset.map(titanic_dataset, lambda x, y: ([x["age"], x["body"], x["embarked"], x["fare"], x["parch"], x["pclass"], x["sex"], x["sibsp"], ], y))

for features, y_true in titanic_dataset.batch(batch_size):
    print(features, y_true)
    with tf.GradientTape as tape:
        network_output = titanic_model(features)
        loss = loss(y_true, network_output)
        print(loss.numpy())

    gradients = tape.gradient(loss, titanic_model.vars)

    # update
    titanic_model.vars.assign_sub(tf.multiply(learning_rate, gradients))