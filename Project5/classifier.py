from conv2d import Conv2d
import tensorflow as tf
from tqdm import trange

class Classifier(tf.Module): 
    def __init__(self, 
    input_depth:int, 
    layer_depths: list[int],
    layer_kernel_sizes: list[tuple[int,int]],
    num_classes:int,
    num_passes: int, 
    hidden_activation=tf.identity,
    output_activation=tf.identity
    #initalize a filter 
    ):
        self.input_depth = input_depth
        self.layer_depths = layer_depths
        self.layer_kernel_sizes = layer_kernel_sizes
        self.num_passes = num_passes 
        self.num_classes = num_classes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        rng = tf.random.get_global_generator()
        k_height = layer_kernel_sizes[0][0]
        k_width  = layer_kernel_sizes[0][1]
        layer_depth = layer_depths[0]
        stride = [1,1,1,1]

        self.infilter = tf.Variable(rng.normal(
            shape=[k_height,k_width,self.input_depth,layer_depth],
            mean=0,
            stddev = tf.sqrt(2/(k_height*k_width*layer_depth))
            ),
            trianable = True
            )
        
        self.hidfilter =  tf.Variable(rng.normal(
            shape=[k_height,k_width,layer_depth,layer_depth],
            mean=0,
            stddev = tf.sqrt(2/(k_height*k_width*layer_depth))
            ),
            trianable = True
            )
        
        self.outfilter = tf.Variable(rng.normal(
            shape=[1,1,layer_depth,self.num_classes],
            mean=0,
            stddev = tf.sqrt(2/(k_height*k_width*layer_depth))
            ),
            trianable = True)
        
        self.inital_conv = Conv2d(self.infilter,stride)
        self.hidden_conv = Conv2d(self.hidfilter,stride)
        self.fc_layer = Conv2d(self.outfilter,stride)

    def __call__(self,x):
            image = self.inital_conv(x)
            for _ in range(self.num_passes): 
                image = self.hidden_activation(self.hidden_conv(image))
            #full layer here 
            image = tf.reduce_mean(image, axis = [1,2], keepdims=True)
            image = self.fc_layer(image)

            return tf.squeeze(image)
    

def main():
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)
    input_depth = 1 
    depths = [1,2,2]
    kernel_size = [(3,3)]
    num_classes = 10 
    num_passes = 9

    model = Classifier(
        input_depth,
        depths,
        kernel_size,
        num_classes,
        num_passes,
        tf.nn.relu,
        tf.nn.softmax
    )