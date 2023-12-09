def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

file = 'CIFAR_batches/train'
train_data = unpickle(file)
x_train = train_data['data']
x_labels = train_data['fine_labels']
x_train = x_train.reshape(len(x_train),3,32,32)
x_train = x_train.transpose(0,2,3,1)
print(x_train.shape)
print(len(x_labels))

