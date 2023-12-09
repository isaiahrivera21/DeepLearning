#!/bin/env python



import tensorflow as tf
from tqdm import trange

from linear import Linear

import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import DecisionBoundaryDisplay

from data import DataSet


def weight_init(shape):
    num_inputs, num_outputs = shape
    weight_vals = tf.random.uniform(shape=[num_inputs, num_outputs])
    # print(weight_vals)
    return weight_vals


class HiddenLayer(tf.Module):
    def __init__(self, out_dim, weight_init=weight_init, activation=tf.identity):
        # Initialize the dimensions and activation functions
        self.out_dim = out_dim
        self.weight_init = weight_init
        self.activation = activation
        self.built = False

    def __call__(self, x):
        if not self.built:
            # Infer the input dimension based on first call
            self.in_dim = x.shape[1]
            self.built = True
            self.w = tf.Variable(self.weight_init(shape=(self.in_dim, self.out_dim)))
            self.b = tf.Variable(tf.zeros(shape=(self.out_dim,)))
        # Compute the forward pass
        # linear = Linear(self.in_dim,self.out_dim)
        z = tf.add(tf.matmul(x, self.w), self.b)
        return self.activation(z)


class MLP(tf.Module):
    def __init__(
        self,
        num_inputs,
        num_outputs,
        num_hidden_layers,
        hidden_layer_width,
        hidden_activation=tf.identity,
        output_activation=tf.identity,
    ):
        # output_activation=tf.identity):
        # idk what we would use number of inputs for????
        # num of inputs??????
        self.input_layer = Linear(num_inputs, hidden_layer_width, bias=True)
        self.hidden_layers = [
            Linear(hidden_layer_width, hidden_layer_width)
            for _ in range(num_hidden_layers)
        ]  # we inilizen an instance of the hidden layer
        self.output_layer = Linear(hidden_layer_width, num_outputs, bias=True)
        self.output_activation = output_activation
        self.num_hidden_layers = num_hidden_layers
        self.hidden_activation = hidden_activation

    # @tf.function
    def __call__(self, x, preds=False):
        # Execute the model's layers sequentially
        current_layer = self.input_layer(x)
        for hidden_layer in self.hidden_layers:
            current_layer = self.hidden_activation(hidden_layer(current_layer))
        return self.output_activation(self.output_layer(current_layer))


def main():
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)
    num_samples = 200
    num_inputs = 2
    batch_size = 128
    num_iters = 1000
    num_inputs = 2
    num_outputs = 1
    num_hidden_layers = 5
    hidden_layer_width = 20

    data_set = DataSet(batch_size)
    model = MLP(
        num_inputs,
        num_outputs,
        num_hidden_layers,
        hidden_layer_width,
        hidden_activation=tf.nn.relu,
        output_activation=tf.nn.sigmoid,
    )

    optimizer = tf.optimizers.Adam(
        learning_rate=0.001, beta_1=0.9, beta_2=0.999
    )  # keeping the default parameters
    bar = trange(num_iters)

    for i in bar:
        with tf.GradientTape() as tape:
            x, y = data_set.batch()
            batch_indices = rng.uniform(
                shape=[batch_size], maxval=num_samples * 2, dtype=tf.int32
            )
            x_batch = tf.gather(x, batch_indices)
            y_batch = tf.reshape((tf.gather(y, batch_indices)), [batch_size, 1])
            # breakpoint()
            y_hat = model(x_batch)
            # print(y_hat)

            loss = tf.math.reduce_mean(
                (
                    (-y_batch * tf.math.log(y_hat + 1e-10))
                    - ((1 - y_batch) * tf.math.log(1 - y_hat + 1e-8))
                )
            )

            l2 = 0.001  # L2 weight decay
            loss = loss - l2
            grads = tape.gradient(loss, model.trainable_variables)
            # breakpoint()
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            bar.set_description(f"Loss @ {i} => {loss.numpy():0.3f}")

    f1, f2 = np.meshgrid(np.linspace(-25.0, 25.0, 600), np.linspace(-25.0, 25.0, 600))
    grid = np.vstack([f1.ravel(), f2.ravel()]).T
    y_pred = np.reshape(model(grid), f1.shape)
    display = DecisionBoundaryDisplay(xx0=f1, xx1=f2, response=y_pred)
    x_a, x_b = data_set.points()

    display.plot()
    display.ax_.scatter(x_a[:, 0], x_a[:, 1])
    display.ax_.scatter(x_b[:, 0], x_b[:, 1])
    plt.savefig("I_tried.pdf")
    # plt.show()


if __name__ == "__main__":
    main()

'''
Citations:

Websites: 
https://www.tensorflow.org/guide/core/mlp_core

Classmates: 

Daniel Tsarev, Jacob Kojziej, Lizelle Ocfemia, Gary Kim, Deigo Toribio

'''


