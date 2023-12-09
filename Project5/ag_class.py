from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import tensorflow as tf 
from transformers import TFAutoModelForSequenceClassification
import numpy as np
from Linear import Linear
from sklearn.metrics import top_k_accuracy_score



batch_size = 128 
num_samples = 50 
num_inputs = 384
num_outputs = 4

dataset = load_dataset('ag_news')

#both are list 
ag_labels = dataset['train']['label']
ag_data = dataset['train']['text']

rng = tf.random.get_global_generator()
rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)
batch_indices = rng.uniform(shape=[batch_size], maxval=num_samples, dtype=tf.int32)
x_batch = tf.gather(ag_data, batch_indices)
x_batch = x_batch.numpy().tolist()
y_batch = tf.gather(ag_labels,batch_indices)

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(x_batch) #probably embed a batch 
# print(embeddings.shape) #have a shape of [batchsize,384]

linear = Linear(num_inputs,num_outputs)

y_hat = linear(embeddings)
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(y_batch,y_hat)
print(loss)
acc = top_k_accuracy_score(y_batch, y_hat, k=1, labels=np.arange(4))
print(acc)
#I have my embeddings 






#want to encode a certain amount of sentenves into vectors 
# batch_indices = rng.uniform(shape=[batch_size], maxval=num_samples, dtype=tf.int32)
# x_batch = tf.gather(ag_data, batch_indices)
# print(x_batch)
# embeddings = model.encode(x_batch)
# print(embeddings.shape)

# print(x_batch.shape) #batch is just 128 

#






# text = ag_data[0]
# labels = ['World', 'Sports', 'Buisness','Sci/Tech']
# result = classifier(text, labels)
# print(result)
# print(text)

