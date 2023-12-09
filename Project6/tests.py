import pytest

"""
d_k --> dimensions of queries and keys 
d_v --> dimensions of values
d_model --> size of the vector that contains the embedded sentence sequence 

"""


def test_mhawork():
    from multihead import MultiHeadAttention
    import tensorflow as tf

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    seq_len = 3
    n_heads = 8
    d_model = 512
    d_k = 64
    d_v = 64
    batch_size = 128

    a = rng.normal(shape=[batch_size, seq_len, d_model])

    multihead = MultiHeadAttention(n_heads, d_model, d_k, d_v)
    expected_shape = [batch_size, seq_len, d_model]
    tf.assert_equal(((multihead(a, a, a, mask=False).shape)), expected_shape)


@pytest.mark.parametrize("d_model", [3, 512, 1024])
def test_block(d_model):
    import tensorflow as tf
    from decoder import DecoderTransformerBlock

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    seq_len = 3
    n_heads = 8
    d_k = 64
    d_v = 64

    a = rng.normal(
        shape=[128, seq_len, d_model]
    )  # batch of sentences of length 3 with each word having 512 dimensions
    expected_shape = [128, seq_len, d_model]

    decoder = DecoderTransformerBlock(n_heads, d_model, d_k, d_v)
    tf.assert_equal(((decoder(a).shape)), expected_shape)


def test_mask():
    import tensorflow as tf
    from multihead import MultiHeadAttention
    import numpy as np

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    seq_len = 3
    n_heads = 8
    d_model = 3
    d_k = 64
    d_v = 64

    a = tf.Variable(rng.normal(shape=[1, seq_len, d_model]), trainable=True)

    mha = MultiHeadAttention(n_heads, d_model, d_k, d_v)

    with tf.GradientTape() as tape:
        tape.watch(a)
        output = mha(a, a, a, mask=True)
    gradients = tape.jacobian(output, a)
    attention_word1 = gradients[0][0][0]

    assert np.any(attention_word1[0][0]) == True
    assert np.any(attention_word1[0][1]) == False
    assert np.any(attention_word1[0][2]) == False
