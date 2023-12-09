import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from numpy import pi
from numpy.random import PCG64, Generator


class DataSet:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.N = 200  # number of samples
        N = self.N
        offset = 4
        theta = np.linspace(0, 3 * pi, N)
        self.rng = Generator(PCG64())
        noise = self.rng.standard_normal()

        r_a = 2 * theta + pi
        data_a = (
            np.array([tf.math.cos(theta) * r_a, tf.math.sin(theta) * r_a]).T + noise
        )
        self.x_a = data_a + np.random.randn(N, 2)

        r_b = -2 * theta - pi
        data_b = (
            np.array([tf.math.cos(theta) * r_b, tf.math.sin(theta) * r_b]).T + noise
        )
        self.x_b = data_b + np.random.randn(N, 2)

        self.x = tf.Variable(
            np.append(self.x_a, self.x_b, axis=0).astype(dtype="float32")
        )
        self.y = tf.Variable(
            np.append(tf.zeros(self.x_a.shape[0]), tf.ones(self.x_b.shape[0])), axis=0
        )

    def batch(self):
        return self.x, self.y

    def plotter(self):
        plt.scatter(self.x_a[:, 0], self.x_a[:, 1])
        plt.scatter(self.x_b[:, 0], self.x_b[:, 1])
        plt.show()

    def points(self):
        return self.x_a, self.x_b


# modified this code: https://gist.github.com/45deg/e731d9e7f478de134def5668324c44c5
