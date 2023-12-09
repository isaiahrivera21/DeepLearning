# if I convolve with different size filters I get the expected shpae that I want
import pytest


# Convolve with different filter sizes
@pytest.mark.parametrize(
    "layer_kernel_sizes", [[(1, 1)], [(3, 3)], [(5, 5)]]
)  # list of tuples
def test_conv(layer_kernel_sizes):
    from conv2d import Conv2d
    import tensorflow as tf

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)
    in_channels = 1
    out_channels = 16

    a = rng.normal(shape=[128, 5, 5, 4])  # 4D tensor
    filter_shape = [
        layer_kernel_sizes[0][0],
        layer_kernel_sizes[0][1],
        in_channels,
        out_channels,
    ]
    filter = tf.random.normal(shape=filter_shape)
    print(filter)
    stride = [1, 1, 1, 1]  # stride of 1
    conv = Conv2d(filter, stride)
    ans = conv(a)
    tf.assert_equal(tf.shape(ans)[-1], out_channels)


# see that group norm generates the same shape as the input
def test_groupnorm():
    from group_norm import GroupNorm
    import tensorflow as tf

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)
    G = 7
    C = 3
    N = 128
    gamma = rng.normal(shape=[1, 1, 1, C])
    beta = rng.normal(shape=[1, 1, 1, C])
    a = rng.normal(shape=[N, 32, 32, C])
    groupnorm = GroupNorm(C, G)
    tf.assert_equal(
        tf.shape(a), tf.shape(groupnorm(a))
    )  # I think the shape shouldn't change when preforming a GN


# can make sure that it works with different group values maybe
def test_resblock():
    # I just want it to work
    from resudual_block import ResidualBlock
    import tensorflow as tf

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    N = 128
    C = 20
    G = 5
    out_channel = 20

    stride = [1, 1, 1, 1]
    filter_shape = [3, 3, C, out_channel]
    filter = rng.normal(shape=filter_shape)

    resblock = ResidualBlock(filter, stride, C, G)


    a = rng.normal(
        shape=[N, 32, 32, C]
    )  # can use an inital conv to change C. Then res block with whatever number for the C channels that I want

    resblock(a)  #
    assert 1 == 1


# Lets see if we get the optimal test shape even though we use different batches
@pytest.mark.parametrize("d_model", [3, 512, 1024])  # list of tuples
def test_Classifier(d_model):
    from classifier import Classifier
    import tensorflow as tf

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)
    input_depth = 3
    layer_depths = [3, 3]
    kernel_size = [(3, 3)]
    num_classes = 10
    num_passes = 3
    N = 128
    C = 3
    G = 5
    out_channel = 20
    hidden = tf.nn.relu
    out = tf.nn.softmax

    a = rng.normal(shape=[batch_size, 32, 32, C])

    model = Classifier(
        input_depth, layer_depths, kernel_size, num_classes, num_passes, G, hidden, out
    )

    expected_shape = [batch_size, num_classes]
    tf.assert_equal(tf.shape(model(a)), expected_shape)
