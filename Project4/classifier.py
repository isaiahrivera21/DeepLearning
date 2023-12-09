from conv2d import Conv2d
from resudual_block import ResidualBlock
from group_norm import GroupNorm
import tensorflow as tf


class Classifier(tf.Module):
    def __init__(
        self,
        input_depth: int,
        layer_depths: list[int],
        layer_kernel_sizes: list[tuple[int, int]],
        num_classes: int,
        num_passes: int,
        G,
        hidden_activation=tf.identity,
        output_activation=tf.identity
        # initalize a filter
    ):
        self.input_depth = input_depth
        self.layer_depths = layer_depths
        self.layer_kernel_sizes = layer_kernel_sizes
        self.num_passes = num_passes
        self.num_classes = num_classes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.G = G
        C = 3

        rng = tf.random.get_global_generator()
        k_height = layer_kernel_sizes[0][0]
        k_width = layer_kernel_sizes[0][1]
        layer_depth = layer_depths[0]
        stride = [1, 1, 1, 1]

        self.infilter = tf.Variable(
            rng.normal(
                shape=[k_height, k_width, self.input_depth, layer_depth],
                mean=0,
                stddev=tf.sqrt(2 / (k_height * k_width * layer_depth)),
            ),
            trianable=True,
            name="infilter",
        )

        self.hidfilter = tf.Variable(
            rng.normal(
                shape=[k_height, k_width, layer_depth, layer_depth],
                mean=0,
                stddev=tf.sqrt(2 / (k_height * k_width * layer_depth)),
            ),
            trianable=True,
            name="hidilter",
        )

        self.outfilter = tf.Variable(
            rng.normal(
                shape=[1, 1, layer_depth, self.num_classes],
                mean=0,
                stddev=tf.sqrt(2 / (k_height * k_width * layer_depth)),
            ),
            trianable=True,
            name="outfilter",
        )

        self.outfilter2 = tf.Variable(
            rng.normal(
                shape=[1, 1, 100, self.num_classes],
                mean=0,
                stddev=tf.sqrt(2 / (k_height * k_width * layer_depth)),
            ),
            trianable=True,
            name="outfilter2",
        )

        self.inital_conv = Conv2d(self.infilter, stride)
        self.hidden_conv = Conv2d(self.hidfilter, stride)
        self.fc_layer = Conv2d(self.outfilter, stride)
        self.fc2_layer = Conv2d(self.outfilter2, stride)
        self.resblock = ResidualBlock(self.hidfilter, stride, C, self.G)
        self.groupnorm = GroupNorm(C, self.G)

    def __call__(self, x):
        image = self.inital_conv(x)
        image = self.groupnorm(image)
        # Pool
        image = tf.nn.relu(image)
        for _ in range(self.num_passes):
            image = self.resblock(image)
        # full layer here

        # setup for the fc layer
        image = tf.reduce_mean(image, axis=[1, 2], keepdims=True)
        image = self.fc_layer(image)
        image = tf.nn.dropout(image, 0.5)
        # breakpoint()
        image = self.fc2_layer(image)

        return tf.squeeze(image)


# what should we change
# maybe we can have an inital res block
# replace the for loop with the res blocks. no hidden acts just res blocks
