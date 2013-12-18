# Create a data set of clusters.  
# Get N clusters with a random number of points each.
# Draw from a normal distribution, each cluster with different mean and variance

import numpy as np
import random as rn
import matplotlib.pyplot as plt
import pickle

N = rn.randint(7,11)

xdata = np.array([])
ydata = np.array([])

for i in range(N):
    
    mean = (rn.uniform(-10,10), rn.uniform(-10,10))
    var = rn.uniform(0,.5)
    cov = [[var,0],[0,var]]
    
    Npoints = rn.randint(50,100)
    newx,newy = np.random.multivariate_normal(mean,cov,Npoints).T

    xdata = np.concatenate((xdata,newx), axis=0)
    ydata = np.concatenate((ydata,newy), axis=0)

   
plt.plot(xdata,ydata,'b.') 
plt.axis([-15, 15, -15, 15])
plt.show()

data = np.vstack(([xdata.T], [ydata.T])).T

pickle.dump([data, N], open('ClusteringDataOneCluster.pkl', 'w'))


