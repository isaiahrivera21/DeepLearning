from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class ImageData:
    def __init__(self, sidelength):
        super().__init__()
        self.sidelength = sidelength

    def __call__(self, img):
        test_img = self.get_testCardF(img)
        self.pixels = tf.reshape(test_img, [-1, 3])
        self.coordinates = self.get_mgrid(self.sidelength)
        return self.coordinates, self.pixels

    def get_testCardF(self, img):
        # convert the image to a tensor
        img = tf.convert_to_tensor(img)
        img = tf.image.resize(img, [self.sidelength, self.sidelength])

        # normalize pixels of an image  [-1,1]
        img = (img / 255.0 - 0.5) / 0.5
        img = tf.reshape(img, shape=[self.sidelength * self.sidelength, 3])
        return img

    def get_mgrid(self, sidelen, dim=2):
        tensors = tuple(dim * [tf.linspace(-1, 1, sidelen)])
        mgrid = tf.stack(tf.meshgrid(*tensors), axis=-1)

        mgrid = tf.reshape(mgrid, (-1, dim))
        return mgrid
