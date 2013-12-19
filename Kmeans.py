# kMeans implentation that creates a N dimensional data set
# and runs a vectorized kMeans algorithm on the data.  Will try different cluster
# numbers.

import numpy as np
import random as rn
import matplotlib.pyplot as plt
from math import sqrt

# functions to implement:

def createDataSet(Nclusters=5, Npoints=100, dims=2, maxMean = 10.0, var = 1.0):
    ''' Nclusters is the number of clusters to simulate
    Each cluster has (0.5,1.5) * Npoints 
    dims is the dimensionality of the data
    Data for each cluster is sampled from a multivariate gaussian 
    (diagonal cov matrix) with mean in [-maxMean, maxMean) with variance in [0+0.5*var,1.5*var)
    Returns an array with the data '''

    Ncluster_points = [rn.randint(0.5*Npoints,1.5*Npoints) for x in range(Nclusters)]
    total_records = sum(Ncluster_points)
    data = np.zeros((total_records,dims))

    Ncounter = 0

    for i in range(Nclusters):  # loop over each cluster

        mean = np.random.uniform(-maxMean, maxMean, dims)
        variance = np.random.uniform(0+0.5*var,1.5*var,1)
        cov = np.identity(dims)*variance
        Num_points_in_cluster = Ncluster_points[i]

        cluster_points = np.random.multivariate_normal(mean,cov,Num_points_in_cluster)
        data[Ncounter:Ncounter + Ncluster_points[i],:] = cluster_points
        Ncounter = Ncounter + Ncluster_points[i]

    return data

def distance(point1, point2):
    ''' Calculates the Euclidean distance between two data points
    '''
    return sqrt(sum(np.square(point1-point2)))

def array_distance(data_array, point):

    distance_array = np.zeros(np.size(data_array,0))

    for index,elem in enumerate(data_array):
        distance_array[index] = distance(elem, point)

    return distance_array

def initializeCentroids(data, Nclusters):
    ''' Initialize the means by randomly selecting Ncluster points from the 
    data without replacement.  Later versions might use smarter methods.'''

    Nrecords = np.size(data,0)
    test_array = range(Nrecords);

    random_sample = np.random.choice(test_array, Nclusters, replace=False)

    return data[random_sample,:]

def findClosestCentroids(data, centroids):
    '''  Given the data and the locations of centroids, calculates the closest
    centroid for each data point.  Returns a vector containing the list of 
    closest centroids. '''

    Nrecords = np.size(data,0)
    C = np.zeros(Nrecords)

    for i in range(Nrecords):
        min_index = -1
        min_distance = None
        point = data[i,:]

        dists = array_distance(centroids, point)
        C[i] = dists.argmin()

    return C


def computeMeans(data, C, centroids):

    new_centroids = np.zeros_like(centroids)
    Ncentroids = np.size(centroids,0)

    for i in range(Ncentroids):
        index = (C == i)
        new_centroids[i,:] = sum(data[index],0)/sum(index)
        # possible error here if no points assigned to centroid - divide by zero
        # not a problem when a lot of points?

    return new_centroids

def evaluateMeanChange(old_centroids, new_centroids):
    ''' Use this function to evaluate convergence.  It gives a scalar that is the sum 
    distance between centroids and new_centroids.  sum_dist should tend to 0 as the 
    kMeans converges.'''

    sum_dist = 0
    for index in range(np.size(old_centroids,0)):
        sum_dist += distance(old_centroids[index,:],new_centroids[index,:])

    return sum_dist

def kMeans(data, NClusterGuess):
    ''' Runs kMeans once. Returns the final centroid positions 
    '''
    
    count = 0
    centroids = initializeCentroids(dataSet, NClusterGuess)


    while True:
        C = findClosestCentroids(dataSet, centroids)
        new_centroids = computeMeans(dataSet, C, centroids)
        change = evaluateMeanChange(centroids, new_centroids)
        print change
        
        plt.hold(True)
        plt.plot(dataSet[:,0], dataSet[:,1], 'bo')
        plt.plot(centroids[:,0], centroids[:,1], 'ro')
        plt.axis('equal')
        plt.show()

        raw_input('Press <ENTER> to continue')

        centroids = new_centroids
        count += 1

        # I should use a % change but this will work for now.
        if ((count > 20) or (change < .0001)): break

    return centroids


dataSet = createDataSet(5,50,2,10,1)
final_centroids = kMeans(dataSet, 5)




'''
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

