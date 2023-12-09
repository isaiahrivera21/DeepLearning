# Write a py test that verifies the data is in the right shape
# https://www.binarystudy.com/2021/09/how-to-load-preprocess-visualize-CIFAR-10-and-CIFAR-100.html

import os
from typing import Any
import numpy as np
import matplotlib.pyplot as plt


class CIFARData:
    def __init__(self, test_val, cifar10_or_cifar100):
        pass

        self.metadata = "CIFAR_batches/batches.meta"
        self.test = "CIFAR_batches/test_batch"
        self.test_val = test_val
        self.choice = cifar10_or_cifar100
        # self.file2 = 'CIFAR_batches/data_batch_2'
        # self.file3 = 'CIFAR_batches/data_batch_3'
        # self.file4 = 'CIFAR_batches/data_batch_4'
        # self.file5 = 'CIFAR_batches/data_batch_5'

    def unpickle(self, file):
        import pickle

        with open(file, "rb") as fo:
            dict = pickle.load(fo, encoding="latin1")
        return dict

    def formatbatch(self, file):
        data_batch = self.unpickle(file)
        image = data_batch["data"]
        image = image.reshape(10000, 3, 32, 32)
        image = image.transpose(0, 2, 3, 1)
        labels = data_batch["labels"]
        return image, labels

    def __call__(self, files):
        batches = []
        if self.test_val == False:
            if self.choice == True:  # cifar10
                for file in files:
                    batch, labels = self.formatbatch(file)
                    batches.append(batch)
                batches = np.reshape(batches, [50000, 32, 32, 3])
                return batches, labels
            if self.choice == False:  # cifar100
                cfile = "CIFAR_batches/train"
                train_data = self.unpickle(cfile)
                x_train = train_data["data"]
                x_labels = train_data["fine_labels"]
                x_train = x_train.reshape(len(x_train), 3, 32, 32)
                x_train = x_train.transpose(0, 2, 3, 1)
                return x_train, x_labels

            # batches = batches.append(batch)
        # labels = self.unpickle(self.metadata)
        # print(batches)

        if self.test_val == True:
            if self.choice == True:
                t_batch, t_label = self.formatbatch(self.test)
            if self.choice == False:
                tfile = "CIFAR_batches/test"
                test_data = self.unpickle(tfile)
                t_batch = test_data["data"]
                t_label = test_data["fine_labels"]
                t_batch = t_batch.reshape(len(t_batch), 3, 32, 32)
                t_batch = t_batch.transpose(0, 2, 3, 1)
        return t_batch, t_label


# file1 = files + "data_batch_1"
# print(files)
# data_batch_1 = unpickle(file1)
# print(data_batch_1.keys())
# datapath = 'CIFAR_batches/'
# files = os.listdir(datapath)


# file1 = 'CIFAR_batches/data_batch_1'
# file2 = 'CIFAR_batches/data_batch_2'
# file3 = 'CIFAR_batches/data_batch_3'
# file4 = 'CIFAR_batches/data_batch_4'
# file5 = 'CIFAR_batches/data_batch_5'
# files = [file1,file2,file3,file4,file5]

# b1 = CIFARData(F)
# batch1, labels1 = b1(files)

# print(labels1)


# print(batch1[0].shape)
# print(labels1['label_names'])
# print(labels1)

# X_train = data_batch_1['data']
# print(X_train)
