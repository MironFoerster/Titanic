import tensorflow as tf
import tensorflow_datasets as tfds
import model
import loss
import data
import matplotlib.pyplot as plt
import os

learning_rate = 0.01
batch_size = 100
num_epochs = 100

titanic_model = model.Model(layer_sizes=[128, 64, 32, 8, 1], input_size=11, activation=tf.tanh)

loss = loss.Loss("mean_squared")
dataset = tfds.load("titanic", split="train", shuffle_files=True, as_supervised=True)
features_removed_train = dataset.map(data.remove_features)
preprocessed = features_removed_train.map(data.preprocess_features)

train = preprocessed.take(900)
test = preprocessed.skip(900)

count = []
train_errors = []
test_errors = []
tps = []
fps = []
tns = []
fns = []
accs = []
senss = []
specs = []
precs = []
negs = []

for e in range(num_epochs):
    epoch_train_error = 0
    for features, y_true in train.batch(batch_size):
        with tf.GradientTape() as tape:
            network_output = titanic_model(features)
            error = loss(y_true, network_output)
            epoch_train_error += error.numpy()
        gradients = tape.gradient(error, titanic_model.vars)

        # update
        for var, grad in zip(titanic_model.vars, gradients):
            var.assign_sub(tf.multiply(learning_rate, grad))

    epoch_test_error = 0
    epoch_tp = 0
    epoch_fp = 0
    epoch_tn = 0
    epoch_fn = 0
    epoch_acc = 0
    epoch_sens = 0
    epoch_spec = 0
    epoch_prec = 0
    epoch_neg = 0
    for features, y_true in test.batch(batch_size, drop_remainder=True):
        network_output = titanic_model(features)
        y_pred = tf.map_fn(lambda x: 2 if x >= 0.5 else 0, network_output)

        s = y_pred + y_true

        tp = tf.reduce_sum(tf.cast(tf.equal(s, 3), dtype=tf.int32), axis=0)
        fp = tf.reduce_sum(tf.cast(tf.equal(s, 2), dtype=tf.int32), axis=0)
        tn = tf.reduce_sum(tf.cast(tf.equal(s, 0), dtype=tf.int32), axis=0)
        fn = tf.reduce_sum(tf.cast(tf.equal(s, 1), dtype=tf.int32), axis=0)

        sens = tp/(tp+fn)
        spec = tn/(tn+fp)
        prec = tp/(tp+fp)
        neg = tn/(tn+fn)

        acc = (tp+tn)/(tp+tn+fp+fn)

        error = loss(y_true, network_output)

        epoch_test_error += error.numpy()
        epoch_tp += tp
        epoch_fp += fp
        epoch_tn += tn
        epoch_fn += fn
        epoch_acc += acc
        epoch_sens += sens
        epoch_spec += spec
        epoch_prec += prec
        epoch_neg += neg

    count.append(e)
    train_errors.append(epoch_train_error/9)
    test_errors.append(epoch_test_error/4)
    tps.append(epoch_tp/4)
    fps.append(epoch_fp/4)
    tns.append(epoch_tn/4)
    fns.append(epoch_fn/4)
    accs.append(epoch_acc/4)
    senss.append(epoch_sens/4)
    specs.append(epoch_spec/4)
    precs.append(epoch_prec/4)
    negs.append(epoch_neg/4)

    print("epoch", e)
    print("train_error:", epoch_train_error / 9)
    print("test_error:", epoch_test_error / 4)
    print("test_acc:", epoch_acc/4)


fig, axs = plt.subplots(2, 2, layout='constrained')
axs[0][0].plot(count, train_errors, label="train")
axs[0][0].plot(count, test_errors, label="test")
axs[0][0].set_title("errors")
axs[0][0].legend()
axs[0][1].plot(count, accs, label="acc")
axs[0][1].set_title("acc")
axs[0][1].legend()
axs[1][0].plot(count, tps, label="tp")
axs[1][0].plot(count, fps, label="fp")
axs[1][0].plot(count, tns, label="tn")
axs[1][0].plot(count, fns, label="fn")
axs[1][0].set_title("tfpn")
axs[1][0].legend()
axs[1][1].plot(count, senss, label="sens")
axs[1][1].plot(count, specs, label="spec")
axs[1][1].plot(count, precs, label="prec")
axs[1][1].plot(count, negs, label="neg")
axs[1][1].set_title("specs")
axs[1][1].legend()

plt.savefig(os.path.join("plots", "plots.svg"))

while True:
    print("Enter features:")
    features = {"age": tf.constant(float(input("Age:")), dtype=tf.float32),
                "embarked": tf.constant(int(input("Embarked:")), dtype=tf.int64),
                "fare": tf.constant(float(input("Fare:")), dtype=tf.float32),
                "parch": tf.constant(float(input("Parch:")), dtype=tf.float32),
                "pclass": tf.constant(int(input("Pclass:")), dtype=tf.int64),
                "sex": tf.constant(float(input("Sex:")), dtype=tf.float32),
                "sibsp": tf.constant(float(input("Sibsp:")), dtype=tf.float32)}

    x = data.preprocess_features(features, tf.constant(0.))
    print(x)
    pred = titanic_model(tf.expand_dims(tf.concat(list(x[0]), axis=0), axis=1))
    print(pred)
