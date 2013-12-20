# kMeans implentation that creates a N dimensional data set
# and runs a vectorized kMeans algorithm on the data.  Will try different cluster
# numbers.

import numpy as np
import random as rn
from math import sqrt

# functions to implement:

def createDataSet(Nclusters=5, Npoints=100, dims=2, maxMean = 10.0, var = 1.0):
    ''' Nclusters is the number of clusters to simulate.
    Each cluster has (0.5,1.5) * Npoints.
    dims is the dimensionality of the data.
    Data for each cluster is sampled from a multivariate gaussian. 
    (diagonal cov matrix) with mean in [-maxMean, maxMean) with variance in [0+0.5*var,1.5*var)
    Returns an array with the data '''

    Ncluster_points = [rn.randint(int(0.5*Npoints),int(1.5*Npoints)) for x in range(Nclusters)]
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
    ''' Given an array of points, returns an array containing the distance between a single
    point and every item of the array.  Uses a loop.'''

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
    ''' Given the data points, the current centroids and the assignments of points to 
    centroids, this computes the new centroids by find the means of the points assigned
    to each centroid.'''

    new_centroids = np.zeros_like(centroids)
    Ncentroids = np.size(centroids,0)

    for i in range(Ncentroids):
        index = (C == i)
        new_centroids[i,:] = sum(data[index],0)/sum(index)
        # possible error here if no points assigned to centroid - divide by zero
        # not a problem when a lot of points?

    return new_centroids

def costFunction(data, C, centroids):
    ''' Calculates the average distance from each point in the data to its assigned 
    centroid.  Should monotonically decrease.  Use it instead of evaluateMeanChange()?'''

    sum = 0
    Npoints = np.size(data,0)

    for i in range(Npoints):
        point = data[i, :]
        centr = centroids[C[i]]
        sum += distance(point, centr)

    return sum/Npoints


def kMeans(data, NClusterGuess):
    ''' Runs kMeans once. Returns the final centroid positions and cost.
    '''
    
    count = 0
    centroids = initializeCentroids(data, NClusterGuess)
    Cinit = findClosestCentroids(data, centroids)
    cost = costFunction(data, Cinit, centroids)

    while True:
        C = findClosestCentroids(data, centroids)
        new_centroids = computeMeans(data, C, centroids)

        new_cost = costFunction(data, C, new_centroids)
        
        centroids = new_centroids
        count += 1

        cost_p_change = (1.0)*(cost-new_cost)/cost
        #print "Cost =", cost
        cost = new_cost

        # I should use a % change but this will work for now.
        if (count > 20) or (cost_p_change < .0000001):
            #print "Cost =", cost
            break

    return centroids, cost
