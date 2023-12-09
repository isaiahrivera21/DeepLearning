from classifier import Classifier
import tensorflow as tf
from tqdm import trange
import yaml
import argparse
from pathlib import Path
from generate_data import CIFARData
import os
from sklearn.metrics import top_k_accuracy_score
import numpy as np
import matplotlib.pyplot as plt


def main():
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)

    datapath = "CIFAR_batches/"
    files = os.listdir(datapath)

    file1 = "CIFAR_batches/data_batch_1"
    file2 = "CIFAR_batches/data_batch_2"
    file3 = "CIFAR_batches/data_batch_3"
    file4 = "CIFAR_batches/data_batch_4"
    file5 = "CIFAR_batches/data_batch_5"
    files = [file1, file2, file3, file4, file5]

    b1 = CIFARData(True, False)
    batch1, labels1 = b1(files)

    parser = argparse.ArgumentParser(
        prog="Linear",
        description="Fits a linear model to some data, given a config",
    )

    parser.add_argument("-c", "--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())

    num_iters = config["learning"]["num_iters"]
    batch_size = config["learning"]["batch_size"]
    num_passes = config["learning"]["num_passes"]

    G = config["model"]["G"]
    input_depth = config["model"]["input_depth"]
    num_classes = config["model"]["num_classes"]

    num_samples = config["data"]["num_samples"]

    kernel_size = [(3, 3)]
    depths = [3]  # worried about this

    model = Classifier(
        input_depth,
        depths,
        kernel_size,
        num_classes,
        num_passes,
        G,
        tf.nn.relu,
        tf.nn.softmax,
    )

    # default Adam learning rate caused the training to become staganant --> nedd a bigger learning rate
    optimizer = tf.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999)
    bar = trange(num_iters)
    # los_vals = []
    # iters = []
    # accur = []

    for i in bar:
        with tf.GradientTape() as tape:
            # x, y = data_set.batch()
            x = tf.Variable(
                ((batch1).astype(dtype="float32")), trainable="False", name="x_batch"
            )
            x = tf.cast(x, tf.float32)

            y = labels1
            y = tf.cast(y, tf.int32)
            batch_indices = rng.uniform(
                shape=[batch_size], maxval=num_samples, dtype=tf.int32
            )
            x_batch = tf.gather(x, batch_indices)
            x_batch = tf.cast(x_batch, tf.float32)
            x_batch = tf.reshape(x_batch, [batch_size, 32, 32, 3])
            x_batch = tf.image.central_crop(x_batch, 0.6)
            x_batch = tf.image.resize_with_crop_or_pad(x_batch, 32, 32)
            x_batch = tf.image.random_flip_left_right(x_batch)
            y_batch = tf.reshape((tf.gather(y, batch_indices)), [batch_size])
            y_hat = model(x_batch)

            # breakpoint()

            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(y_batch, y_hat)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            lossy = tf.reduce_mean(loss)
            lossy = lossy + 0.01

            acc = top_k_accuracy_score(y_batch, y_hat, k=5, labels=np.arange(100))
            bar.set_description(
                f"Loss @ {i} => {lossy:0.3f}, Accuracy @ {i} => {acc:0.3f}"
            )

            # los_vals.append(lossy)
            # iters.append(i)
            # accur.append(acc)

    # plt.subplot(121)
    # plt.plot(iters, los_vals, "-")
    # plt.title("Loss")
    # plt.xlabel("iters")
    # plt.ylabel("loss")

    # plt.subplot(122)
    # plt.plot(iters, accur, "-")
    # plt.title("Accuracy")
    # plt.xlabel("iters")
    # plt.ylabel("accuracy")

    # plt.suptitle("Charting Loss and Accuracy")
    # plt.show()

if __name__ == "__main__":
    main()

# All citations for papers used are in writeup
# Classmate Citations: Lizelle Ocfemia 

