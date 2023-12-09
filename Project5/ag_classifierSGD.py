from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import tensorflow as tf 
from transformers import TFAutoModelForSequenceClassification
import numpy as np
from Linear import Linear
from sklearn.metrics import top_k_accuracy_score
from tqdm import trange

if __name__ == "__main__":

    batch_size = 128 
    num_samples = 384
    num_inputs = 384
    num_outputs = 4
    num_iters = 2000

    dataset = load_dataset('ag_news')

    #both are list 
    ag_labels = dataset['train']['label']
    ag_data = dataset['train']['text']

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)

    bar = trange(num_iters)
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    linear = Linear(num_inputs,num_outputs)
    optimizer = tf.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

    for i in bar:
        batch_indices = rng.uniform(shape=[batch_size], maxval=num_samples, dtype=tf.int32)
        with tf.GradientTape() as tape:
            x_batch = tf.gather(ag_data, batch_indices)
            x_batch = x_batch.numpy().tolist()
            y_batch = tf.gather(ag_labels,batch_indices)
            embeddings = model.encode(x_batch) #probably embed a batch 
            y_hat = linear(embeddings)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(y_batch,y_hat)
            lossy = tf.reduce_mean(loss)
            acc = top_k_accuracy_score(y_batch, y_hat, k=1, labels=np.arange(4))
            bar.set_description(
                f"Loss @ {i} => {lossy:0.3f}, Accuracy @ {i} => {acc:0.3f}"
            )
    #I have my embeddings 