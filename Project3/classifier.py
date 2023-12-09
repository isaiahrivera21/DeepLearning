import tensorflow as tf 
from conv2d import Conv2d
from mlp import MLP

class Classifier(tf.Module):
    def __init__(
        self, 
        input_depth:int, # number of channels 
        layer_depths: list[int], #<--
        layer_kernel_sizes: list[tuple[int,int]], #kernel dimensions 
        num_classes:int, 
        num_passes: int, #number of convolutions  
        hidden_activation=tf.identity,
        output_activation=tf.identity
        #initalize a filter 
        ):
        # self.inp = num_passes
        self.rng = tf.random.get_global_generator()
        self.input_depth = input_depth
        self.layer_depths = layer_depths
        self.layer_kernel_sizes = layer_kernel_sizes
        self.num_classes = num_classes
        self.num_passes = num_passes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        strides = [1,1,1,1]

        conv_filter = self.generate_filter(3,self.input_depth,self.num_classes)
        #could generate 3 filters 

        self.infilter = tf.Variable(self.rng.normal(
            shape=[3,3,self.input_depth,10],
            mean=0,
            stddev = tf.sqrt(2/(3*3*1))
            ),
            trianable = True
            )


        self.conv2d = Conv2d(conv_filter,strides)
        self.fc_layer = MLP(
            num_inputs=self.input_depth,
            num_outputs = self.num_classes,
            num_hidden_layers= 2,
            hidden_layer_width=640,#??? 
            hidden_activation=tf.nn.relu, 
            output_activation=tf.nn.softmax
        )

    
    def generate_filter(self,layer_filter_size,filter_depth,out_channel):
        n_hat = layer_filter_size * layer_filter_size *  filter_depth #k * k * d 
        stddev = tf.sqrt (2 / n_hat)
        # FROM: Paper
        mean = 0 

        filter = tf.Variable(
            self.rng.normal(shape=[layer_filter_size,layer_filter_size,filter_depth,out_channel], mean=mean, stddev=stddev),
            trainable = True 
        )
        #use optimal initalizer found in paper 
        # print(filter)

        return filter 
    
    def flatten(self,tensor):
        flat_tensor = tf.reshape(tensor,shape=[-1,tensor.shape[1]*tensor.shape[2]*tensor.shape[3]])
        return flat_tensor
    
    def __call__(self,x):
        #inital convolution? 
        image = self.conv2d(x)
        image = self.conv2d(image)
        print(image.shape)
        for _ in range(self.num_passes): 
            x = self.hidden_activation(self.conv2d(x))
            print(x.shape)
        #full layer here 
        x_flat = self.flatten(x)
        return self.fc_layer(x_flat)








