import tensorflow as tf

class Conv2d(tf.Module):
    def __init__(self,filters,strides):
        self.filters = filters 
        self.strides = strides 
    def __call__(self,x):
        return tf.nn.conv2d(x,self.filters,self.strides,padding='VALID')