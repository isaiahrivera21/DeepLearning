#if I convolve with different size filters I get the expected shpae that I want 
import pytest

#Convolve with different filter sizes 
@pytest.mark.parametrize("layer_kernel_sizes", [[(1,1)], [(3,3)], [(5,5)]]) #list of tuples 
def test_conv (layer_kernel_sizes):
    from conv2d import Conv2d
    import tensorflow as tf 

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)
    in_channels = 1
    out_channels = 16

    a = rng.normal(shape=[128,5,5,4]) #4D tensor
    filter_shape = [layer_kernel_sizes[0][0],layer_kernel_sizes[0][1],in_channels,out_channels]
    filter = tf.random.normal(shape=filter_shape)
    print(filter)
    stride = [1,1,1,1] #stride of 1
    conv = Conv2d(filter,stride)
    ans = conv(a)
    tf.assert_equal(tf.shape(ans)[-1], out_channels)

#can write a test making sure filter generates the correct shape 
# @pytest.mark.parametrize("layer_kernel_sizes,kernel_depth", [([(1,1)],3), ([(3,3)],4), ([(5,5)],1)]) #list of tuples 
# def test_filter_shape(layer_kernel_sizes,kernel_depth):
#     from classifier import Classifier
#     clasifier = Classifier(

#     )


#we expect to get something of batch_size,out_channels
# def test_final_shape():
#     #take classifier and compare the output to something 
#     expected_shape = [128,10]

#     pass 



# def main():
#     layer = [(3,3)]
#     print(layer[0][1])
#     conv_test(layer)
  
  
# # Using the special variable 
# # __name__
# if __name__=="__main__":
#     main()

    
    