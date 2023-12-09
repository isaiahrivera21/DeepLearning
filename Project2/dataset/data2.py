import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from numpy.random import Generator, PCG64

class DataSet():
    def __init__(self,batch_size): 
        self.batch_size = batch_size
        self.N = 200 #number of samples 
        N = self.N
        theta = np.sqrt((np.random.rand(N)))*3*pi # np.linspace(0,2*pi,100)

        self.rng = Generator(PCG64())
        noise = self.rng.standard_normal() 
        self.offset = 35

        r_a = 2*theta + pi # * noise
        data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
        self.x_a = data_a + np.random.randn(N,2)

        self.x_a = self.x_a + self.offset


        r_b = -2*theta - pi # *noise
        data_b = np.array([np.cos(theta)*r_b, np.sin(theta)*r_b]).T
        self.x_b = data_b + np.random.randn(N,2)

        self.b_a = self.b_a + self.offset

        self.x = np.append(self.x_a[:,0],self.x_b[:,0])
        print(type(self.x))
        self.y = np.append(self.x_a[:,1],self.x_b[:,1])
        

    def batch(self): 
        self.batch_size
        a = np.arange(self.N)
        mini_batch = self.rng.choice(a,self.batch_size)
        return self.x[mini_batch],self.y[mini_batch]

    def plotter(self):
        plt.scatter(self.x_a[:,0],self.x_a[:,1])
        plt.scatter(self.x_b[:,0],self.x_b[:,1])
        plt.show()

def main():
    data = DataSet(100)
    x,y = data.batch()
    data.plotter()





  
  
# Using the special variable 
# __name__
if __name__=="__main__":
    main()

# np.savetxt("result.csv", res, delimiter=",", header="x,y,label", comments="", fmt='%.5f')



#modified this code: https://gist.github.com/45deg/e731d9e7f478de134def5668324c44c5 

