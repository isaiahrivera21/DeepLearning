# Scaled Dot product attention using keras and einops

from typing import Any
import tensorflow as tf
from linear import Linear
from einops import rearrange
from einops import einsum
import math
import numpy as np


class MultiHeadAttention(tf.Module):
    def __init__(self, n_heads, d_model, d_k, d_v):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k

        """
        Instead of performing a single attention function with
        d_model-dimensional keys, values and queries,
        we found it beneficial to linearly project the queries,
        keys and values h times with different, learned
        linear projections to d_k, d_k and d_v dimensions, respectively
        """
        # paramter matricies (projections)
        self.W_q = Linear(d_model, n_heads * d_k, bias=False)
        self.W_k = Linear(d_model, n_heads * d_k, bias=False)
        self.W_v = Linear(d_model, n_heads * d_v, bias=False)

        self.W_0 = Linear(n_heads * d_v, d_model)

    def __call__(self, q, k, v, mask=True):
        # Think we are splitting into n_head matricies
        Q = rearrange(self.W_q(q), "b l (h k) -> h b l k", h=self.n_heads)
        K = rearrange(self.W_k(k), "b t (h k) -> h b t k", h=self.n_heads)
        V = rearrange(self.W_v(v), "b t (h v) -> h b t v", h=self.n_heads)
        attention = einsum(Q, K, "h b l k,h b t k->h b l t") / np.sqrt(self.d_k)
        if mask is True:
            attention = self.fill_mask(attention)
        attention = tf.nn.softmax(attention)
        out = einsum(attention, V, "h b l t ,h b t v->h b l v")
        out = rearrange(out, "h b l v -> b l (h v)")
        out = self.W_0(out)  # output should be [b l d_model]
        return out

    def fill_mask(self, matrix):
        matrix_shape = matrix.shape
        mask = np.zeros(shape=matrix_shape)
        for i in range(matrix_shape[-1]):
            mask[:, :, i, 0 : i + 1] = 1
        negmask = 1 - mask
        num = 3.4 * math.pow(10, 38)

        M = matrix * mask  # mask out things we  want to kill
        N = -((negmask * num + num) - num)  # makes extreemly high values
        matrix = M + N  # <-- makes the false values equal to a huge number
        return matrix
