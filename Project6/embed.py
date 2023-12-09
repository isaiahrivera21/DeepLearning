from typing import Any
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math


class Embeddings:
    def __init__(self, d_model):
        self.d_model = d_model

    def __call__(self, sentence):
        tokens = self.tokenize_sentence(sentence)
        vocab = self.build_vocab(tokens)
        return self.embed_vocab(len(vocab), self.d_model), vocab

    def tokenize_sentence(self, sentence):
        for punc in ["!", ".", "?"]:
            sentence = sentence.replace(punc, "")

        tokens = [token.lower() for token in sentence.split(" ")]

        return tokens

    def build_vocab(self, tokens, speacialTokens=True):
        if speacialTokens:
            tokens = ["<start>"] + tokens + ["<end>"] + ["<pad>"]
        keys = tokens
        vals = np.arange(len(tokens))
        vals = vals.tolist()
        vocab = dict(zip(keys, vals))
        return vocab

    def embed_vocab(self, vocab_size, d_model):
        rng = tf.random.get_global_generator()
        rng.reset_from_seed(2384230948)
        embeddings = rng.normal(shape=[vocab_size, d_model])
        return embeddings
