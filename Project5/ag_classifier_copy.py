from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import tensorflow as tf 
import numpy as np
from Linear import Linear
from sklearn.metrics import top_k_accuracy_score
from tqdm import trange
from classifier import Classifier

if __name__ == "__main__":
    train_samples = 112400
    batch_size = 512
    num_samples = 7600
    num_inputs = 384
    num_outputs = 4
    num_iters = 1000

    dataset = load_dataset('ag_news')

    #both are list 
    ag_labels = dataset['train']['label']
    ag_data = dataset['train']['text']
    test_labels = dataset['test']['label']
    test_data = dataset['test']['text']

    #splitting into train and validation
    ag_val_data = ag_data[:7600] 
    ag_train_data = ag_data[7600:]
    ag_val_labels = ag_labels[:7600]
    ag_train_labels = ag_labels[7600:]

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)

    bar = trange(num_iters)
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    linear = Linear(num_inputs,num_outputs,bias=False)
    optimizer = tf.optimizers.Adam(learning_rate=0.3, beta_1=0.09, beta_2=0.88)

    input_depth = 1 
    depths = [3]
    kernel_size = [(3,3)]
    num_classes = 4 
    num_passes = 3

    classify = Classifier(
        input_depth,
        depths,
        kernel_size,
        num_classes,
        num_passes,
        tf.nn.relu,
        tf.nn.sigmoid
    )

    for i in bar:
        batch_indices = rng.uniform(shape=[batch_size], maxval=train_samples, dtype=tf.int32)
        with tf.GradientTape() as tape:
            x_batch = tf.gather(ag_train_data, batch_indices)
            x_batch = x_batch.numpy().tolist()
            y_batch = tf.gather(ag_labels,batch_indices)
            embeddings = model.encode(x_batch) #probably embed a batch 
            # embeddings.reshape(batch_size,num_inputs,1,1)
            emb = tf.Variable(embeddings, trainable=False)
            emb = tf.reshape(emb,shape=[batch_size,num_inputs,1,1])
            y_hat = classify(emb)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(y_batch,y_hat)
            # l2 = .01 * tf.reduce_sum(tf.square(loss)) #what goes in for x 
            # loss = l2 + loss
            lossy = tf.reduce_mean(loss)
            acc = top_k_accuracy_score(y_batch, y_hat, k=1, labels=np.arange(4))
            bar.set_description(
                f"Loss @ {i} => {lossy:0.3f}, Accuracy @ {i} => {acc:0.3f}"
            )