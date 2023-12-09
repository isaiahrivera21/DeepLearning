from classifier import Classifier
import tensorflow as tf 
import numpy as np 
from tqdm import trange
from MNIST_data import data_dic

def main():
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)
    num_samples = 200
    num_inputs = 2
    input_depth = 1 
    batch_size = 128
    num_iters = 1000
    num_inputs = 2
    num_outputs = 1
    num_hidden_layers = 5
    hidden_layer_width = 20
    depths = [10]
    kernel_size = [(3,3)]
    num_classes = 10 
    num_passes = 10

    model = Classifier(
        input_depth,
        depths,
        kernel_size,
        num_classes, 
        num_passes,
        hidden_activation=tf.nn.relu,
        output_activation=tf.nn.softmax)
    
    optimizer = tf.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)  # keeping the default parameters
    bar = trange(num_iters)

    for i in bar:
        with tf.GradientTape() as tape:
            print(i)
            x = tf.Variable(((data_dic['train_images']).astype(dtype="float32")), trainable = 'False')
            x = tf.cast(x,tf.float32)
            y = (data_dic['train_labels'])
            y = tf.cast(y,tf.int32)
            batch_indices = rng.uniform(
                shape=[batch_size], maxval=num_samples, dtype=tf.int32
            )
            x_batch = tf.gather(x, batch_indices)
            x_batch = tf.cast(x_batch,tf.float32)
            x_batch = tf.reshape(x_batch, [batch_size,28,28,1])
            y_batch = tf.reshape((tf.gather(y, batch_indices)), [batch_size])
            print(x_batch.shape)
            y_hat = model(x_batch)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(y_batch,y_hat)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            bar.set_description(f"Loss @ {i} => {loss.numpy():0.3f}")
    


    print(loss)
    # print(y_batch)

    
    # labels = y_batch
    # logits = y_hat


    
    # print(loss)
    



if __name__ == "__main__":
    main()