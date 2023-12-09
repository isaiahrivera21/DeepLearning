#just play around with different types of augmentations and lets see what they do

import matplotlib.pyplot as plt 
import tensorflow as tf 
from generate_data import CIFARData
import numpy as np 

file1 = 'CIFAR_batches/data_batch_1'
file2 = 'CIFAR_batches/data_batch_2'
file3 = 'CIFAR_batches/data_batch_3'
file4 = 'CIFAR_batches/data_batch_4'
file5 = 'CIFAR_batches/data_batch_5'
files = [file1,file2,file3,file4,file5]
batch_size = 128
b1 = CIFARData()
batch1, labels1 = b1(files)

print(batch1[0].shape) 
print(type(batch1[0]))

rng = tf.random.get_global_generator()
rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)
num_boxes = 1
size = (32,32)

img = tf.Variable(batch1[3], trainable=True )

newimg = tf.image.central_crop(img,0.4)

x = tf.Variable(((batch1).astype(dtype="float32")), trainable = 'False')
x = tf.cast(x,tf.float32)
y = (labels1)
y = tf.cast(y,tf.int32)
batch_indices = rng.uniform(
    shape=[batch_size], maxval=50, dtype=tf.int32
)
x_batch = tf.gather(x, batch_indices)
x_batch = tf.cast(x_batch,tf.float32)
x_batch = tf.reshape(x_batch, [batch_size,32,32,3])
y_batch = tf.reshape((tf.gather(y, batch_indices)), [batch_size])

 
x_batch = tf.image.central_crop(x_batch,0.6)
x_batch = tf.image.resize_with_crop_or_pad(x_batch,32,32)
x_batch = tf.image.random_flip_left_right(x_batch)






