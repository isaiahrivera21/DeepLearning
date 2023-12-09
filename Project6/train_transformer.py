import tensorflow as tf
from transformer import DecoderTransformer
from embed import Embeddings
from tqdm import trange


def main():
    num_iters = 800
    seq_len = 5
    n_heads = 8
    d_model = 3
    d_k = 64
    d_v = 64
    N_x = 8
    sentence = "man bites dog"

    embed = Embeddings(d_model)
    optimizer = tf.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

    [vocab_embeddings, vocabulary] = embed(sentence)
    model = DecoderTransformer(N_x, n_heads, d_model, d_k, d_v, len(vocabulary))

    seq = "<start> man bites dog <end>"
    target = "man bites dog <end> <pad>"

    sequence_position = [vocabulary[word] for word in embed.tokenize_sentence(seq)]
    target_postion = [vocabulary[word] for word in embed.tokenize_sentence(target)]

    bar = trange(num_iters)
    for i in bar:
        with tf.GradientTape() as tape:
            x = tf.gather(vocab_embeddings, sequence_position)
            x_batch = tf.reshape(x, shape=[1, seq_len, d_model])
            y = target_postion
            y_hat = model(x_batch, mask=True)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=y, logits=tf.squeeze(y_hat)
            )
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            index = tf.math.argmax(y_hat, axis=-1)

        bar.set_description(f"Loss @ {i} => {tf.reduce_mean(loss):0.4f}")
        bar.refresh()

    print("TARGET:   ", target_postion)
    tf.print("Predicted:", index, summarize=-1)
    # positional embeddings are the index numbers b/c we are using a VERY small vocab


if __name__ == "__main__":
    main()


"""
Citations: 
https://medium.com/@hunter-j-phillips/the-embedding-layer-27d9c980d124
https://einops.rocks/pytorch-examples.html
https://arxiv.org/abs/1706.03762
https://stackoverflow.com/questions/47447272/does-tensorflow-have-the-function-similar-to-pytorchs-masked-fill
"""
