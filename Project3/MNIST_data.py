#take the MNIST data set and convert it to numpy array 
#Citation: https://www.youtube.com/watch?v=6xar6bxD80g&t=1730s 

import os
import numpy as np 
import codecs 

datapath = 'MNIST_Data/'
files = os.listdir(datapath)
def get_int(b):
    return int(codecs.encode(b,'hex'),16)

# will be in the call function 
data_dic = {}
for file in files:
    if file.endswith('ubyte'):
        #print('Reading',file)
        with open(datapath+file,'rb') as f:
            data = f.read()
            type = get_int(data[:4])
            length = get_int(data[4:8])
            if(type == 2051):
                category = 'images'
                num_rows = get_int(data[8:12])
                num_cols = get_int(data[12:16])
                parsed = np.frombuffer(data,dtype=np.uint8,offset=16)
                parsed = parsed.reshape(length,num_rows,num_cols)
            elif(type == 2049):
                category = 'labels'
                parsed = np.frombuffer(data,dtype=np.uint8,offset=8)
                parsed = parsed.reshape(length)
            if(length==10000):
                set = 'test'
            elif(length==60000): 
                set = 'train'
            data_dic[set+'_'+category] = parsed
#have a numpy array that I can cast into a tensor! 
number = (data_dic['train_images'][0,:,:]).astype(dtype=np.float32)
# print(number.shape) #28 by 28 # 
#in_channel = 1 


#think filters are just weights but mulltidimesniaonal 

#first layer = Conv Layer 
#input = activation(2dConv(input,filter)) <--- CNN layers 
#input pooling(input) #tf.nn.pool
#do this a few times 
#uhhh flattening ???? tf.nest.flatten --> does not work on tensors or numpy arrays??? ?
#fully connected ???? --> linear module 
#tf.nn.sparse_softmax_cross_entropy_with_logits
#flatten could mayber be tensor.reshape ?? 
            



