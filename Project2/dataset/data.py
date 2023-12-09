import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from numpy.random import Generator, PCG64

def data_set():
    N = 200
    offset = 4
    # theta = (np.random.rand(N))*2*pi 
    theta = np.linspace(0,3*pi,N)

    rng = Generator(PCG64())
    noise = rng.standard_normal()

    r_a = 2*theta + pi #* noise
    data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T + noise
    x_a = noise + data_a  + np.random.randn(N,2)
    x_a = x_a + offset

    r_b = -2*theta - pi # *noise
    data_b = np.array([np.cos(theta)*r_b, np.sin(theta)*r_b]).T + noise
    x_b = noise + data_b + np.random.randn(N,2)

    x_b = x_b + offset
    print(x_b)


    res_a = np.append(x_a, np.zeros((N,1)), axis=1)
    res_b = np.append(x_b, np.ones((N,1)), axis=1)

    res = np.append(res_a, res_b, axis=0)
    # print(res_a) #given an xy point is it a one or 0 aka red or blue 
    np.random.shuffle(res)


    # X = np.reshape(X,[])
    Y = np.append(x_a[:,1],x_b[:,1]).astype(dtype='float32')


    # np.savetxt("result.csv", res, delimiter=",", header="x,y,label", comments="", fmt='%.5f')

    plt.scatter(x_a[:,0],x_a[:,1]) #spiral 1
    plt.scatter(x_b[:,0],x_b[:,1]) #spiral 2 
    plt.show()

    return X, Y



#modified this code: https://gist.github.com/45deg/e731d9e7f478de134def5668324c44c5 

