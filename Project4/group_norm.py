import tensorflow as tf
from generate_data import CIFARData


# x --> inout features
# gamma,beta --> scale and offset
# G --> number of group norms
class GroupNorm(tf.Module):
    def __init__(self, C, G):
        # intailize gamma and beta to train
        rng = tf.random.get_global_generator()
        self.gamma = tf.Variable(
            rng.normal(shape=[1, 1, 1, C]), trainable=True, name="Groupnorm/mu"
        )
        self.beta = tf.Variable(
            # .1 * tf.ones(self.M),
            rng.normal(shape=[1, 1, 1, C]),
            trainable=True,
            name="Basis/sigma",
        )

        self.C = C
        self.G = G

    def __call__(self, x, eps=1e-5):
        N, H, W, C = x.shape  # [batch_size,Height,Width,Channels]
        self.G = min(self.G, C)
        # breakpoint()
        x = tf.reshape(x, [N, self.G, C // self.G, H, W])  # <--ERROR

        mean, var = tf.nn.moments(x, [2, 3, 4], keepdims=True)
        x = (x - mean) / tf.sqrt(var + eps)

        x = tf.reshape(x, [N, H, W, C])  # Lets see if this causes an error
        return x * self.gamma + self.beta
        # C has to divible by C


# def main():

#     gamma = 12
#     beta = 12
#     batch_size = 128
#     file1 = 'CIFAR_batches/data_batch_1'
#     file2 = 'CIFAR_batches/data_batch_2'
#     file3 = 'CIFAR_batches/data_batch_3'
#     file4 = 'CIFAR_batches/data_batch_4'
#     file5 = 'CIFAR_batches/data_batch_5'
#     files = [file1,file2,file3,file4,file5]
#     b1 = CIFARData()
#     x, labels1 = b1(files)
#     rng = tf.random.get_global_generator()
#     rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)
#     batch_indices = rng.uniform(
#                 shape=[batch_size], maxval=50, dtype=tf.int32
#             )
#     x_batch = tf.gather(x, batch_indices)
#     x_batch = tf.cast(x_batch,tf.float32)
#     x_batch = tf.reshape(x_batch, [batch_size,32,32,3])
#     print(x_batch.shape)
#     gn = GroupNorm(3,5)
#     ds = gn(x_batch)
#     print(ds.shape)


# Using the special variable
# __name__
# if __name__=="__main__":
#     main()
