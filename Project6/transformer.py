# transformer (decoder only model)
from typing import Any
import tensorflow as tf
from decoder import DecoderTransformerBlock
from linear import Linear


class DecoderTransformer(tf.Module):
    def __init__(self, Nx, n_heads, d_model, d_k, d_v, vocab_size):
        self.first_block = DecoderTransformerBlock(n_heads, d_model, d_k, d_v)
        self.blocks = [
            DecoderTransformerBlock(n_heads, d_model, d_k, d_v) for _ in range(Nx - 1)
        ]
        self.linear = Linear(d_model, vocab_size, bias=False)

    def __call__(self, X, mask):
        self.current_block = self.first_block(X)
        for block in self.blocks:
            self.current_block = block(self.current_block, mask=mask)

        return tf.nn.softmax(self.linear(self.current_block))
