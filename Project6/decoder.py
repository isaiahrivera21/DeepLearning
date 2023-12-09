from typing import Any
from multihead import MultiHeadAttention
import tensorflow as tf
from linear import Linear


class DecoderTransformerBlock:
    def __init__(self, n_heads, d_model, d_k, d_v):
        self.layernorm = tf.keras.layers.LayerNormalization(axis=1)
        self.multihead = MultiHeadAttention(n_heads, d_model, d_k, d_v)
        self.linear1 = Linear(d_model, n_heads * d_v)
        self.hidden_layers = [Linear(n_heads * d_v, n_heads * d_v) for _ in range(4)]
        self.linear2 = Linear(n_heads * d_v, d_model)

    def __call__(self, X, mask=False):
        mha = self.multihead(X, X, X, mask=mask)
        F = self.layernorm(X + mha)
        feed_foward = self.linear1(F)
        for hidden_layer in self.hidden_layers:
            feed_foward = tf.nn.relu(hidden_layer(feed_foward))
        feed_foward = self.linear2(feed_foward)

        out = self.layernorm(F + feed_foward)
        return out
