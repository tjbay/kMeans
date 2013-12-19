# kMeans implentation that creates a N dimensional data set
# and runs a vectorized kMeans algorithm on the data.  Will try different cluster
# numbers.

import numpy as np
import random as rn
import matplotlib.pyplot as plt
import pickle
from math import sqrt

# functions to implement:

def createDataSet(Nclusters=5, Npoints=100, dims=2, maxMean = 10, var = 1):
    ''' Nclusters is the number of clusters to simulate
    Each cluster has (0.5,1.5) * Npoints 
    dims is the dimensionality of the data
    Data for each cluster is sampled from a multivariate gaussian 
    (diagonal cov matrix) with mean in [-maxMean, maxMean) with variance in [0+var/10,var)
    Returns an array with the data '''

    Ncluster_points = [rn.randint(0.5*Npoints,1.5*Npoints) for x in range(Nclusters)]
    total_records = sum(Ncluster_points)
    data = np.zeros((total_records,dims))

    Ncounter = 0

    for i in range(Nclusters):  # loop over each cluster

        mean = np.random.uniform(-maxMean, maxMean, dims)
        variance = np.random.uniform(0+var/10,var,1)
        cov = np.identity(dims)*variance
        Num_points_in_cluster = Ncluster_points[i]

        cluster_points = np.random.multivariate_normal(mean,cov,Num_points_in_cluster)
        data[Ncounter:Ncounter+Ncluster_points[i],:] = cluster_points
        Ncounter = Ncounter + Ncluster_points[i]

    return data

def distance(point1, point2):
    ''' Calculates the Euclidean distance between two data points
    '''
    return sqrt(np.dot(point1, point2))

def initializeCentroids(data, Nclusters):
    ''' Initialize the means by randomly selecting Ncluster points from the 
    data without replacement.  Later versions might use smarter methods.'''

    Nrecords = np.size(data,0)
    test_array = range(Nrecords);

    random_sample = np.random.choice(test_array, Nclusters, replace=False)

    return data[random_sample,:]




dataSet = createDataSet(5,100,5,10,1)
print dataSet

#def initializeCentroids():
#def findClosestCentroids():
#def computeMeans():
#def kMeans():
#def evaluateFit():
'''


def change(means, new_means):
    sum_d = 0
    for index in range(len(means)):
        sum_d += distance(means[index], new_means[index])
    return sum_d
    
def plot_clusters(data, means):
    x_val = [x[0] for x in data]
    y_val = [x[1] for x in data]

    x_means = [x[0] for x in means]
    y_means = [x[1] for x in means]

    plt.plot(x_val,y_val,'b.',x_means, y_means, 'ro') 
    plt.axis([-15, 15, -15, 15])
    plt.show()

def find_closest_cluster(data, means):
    for x in data:
        min_index = -1
        min_distance = 9999
        for index, c_point in enumerate(means):
            if distance(x[0:2], c_point) < min_distance:
                min_distance = distance(x[0:2], c_point)
                min_index = index
                x[2] = min_index

def find_new_cluster_points(data, means, new_means):
    for x in data:
        new_means[x[2]][0] += x[0]
        new_means[x[2]][1] += x[1]
        new_means[x[2]][2] += 1

    for index, cluster in enumerate(new_means):
        if cluster[2] > 0:
            cluster[0] = cluster[0]/cluster[2]
            cluster[1] = cluster[1]/cluster[2]
        else:
            cluster[0:2] = means[index]


            

def Kmeans(data, NumC):
    count = 0
    go = True

    means = np.array([ [0.0,0.0] for i in range(NumC)])
    initialize_means(data, NumC, means)

    #print 'Initial Clusters'
    #print 'Num Clusters:', NumC
    #plot_clusters(data, means)
    
    while go == True:
        count += 1
        new_means = np.array([ [0.0,0.0,0.0] for items in means])

        find_closest_cluster(data, means)
        find_new_cluster_points(data, means, new_means)

        distance  = change(means, new_means[:, [0,1]])
        
        if distance < .0001: go = False
        if count > 20: go = False

        means = new_means[:, [0,1]]

        if count%3==5:
            plot_clusters(data, means)
            print 'Iters:', count
            print 'Update distance =', distance

    #print 'Final Clusters'
    #plot_clusters(data, means)
    #print 'Iters:', count
    #print 'Update distance =', distance
    
    return means

def EvaluateFit(data, means):
    tot_distance = 0
    for index, point in enumerate(means):
        for datapoint in data:
            if datapoint[2] == index:
                tot_distance += distance(point, datapoint[0:2])

    return tot_distance
    
# Load cluster data from pickle file
[data, N] = pickle.load(open('ClusteringDataLowVar.pkl','r'))
data = np.insert(data, 2, values=-1, axis=1)  # add an extra column for cluster ID
np.random.shuffle(data) # Shuffle for no good reason except debugging

# Cluster numbers to test
Nrange = range(3,25)

# Number of times to run the average for each number of clusters
average_num = 10

average_fit = [0 for i in Nrange]

for index, NumC in enumerate(Nrange):
    print 'Testing Number of Clusters =', NumC
    fit_sum = 0
    for i in range(average_num):
        cluster_result = Kmeans(data, NumC)
        fit_sum += EvaluateFit(data, cluster_result)
    average_fit[index] = fit_sum/average_num
        
plt.plot(Nrange, average_fit, 'bo')
plt.xlabel('Number of clusters', fontsize=15)
plt.ylabel('Average Fit Error(lower is better)', fontsize=15)
plt.show()

'''

