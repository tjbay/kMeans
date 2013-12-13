# assumes 2D data

import numpy as np
import random as rn
import matplotlib.pyplot as plt
import pickle
from math import sqrt


def distance(point1, point2):
    #Calculate Euclidean distance between two vectors
    if len(point1) != len(point2):
        print 'Distance: Not the same dimensionality'
        return -1
    
    sum = 0
    for i in range(len(point1)):
        sum += (point1[i]-point2[i])**2

    return sqrt(sum)

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

def initialize_means(data, NumC, means):
    
    while len(np.unique(means)) < len(means):
        for x in means:
            temp = rn.randint(0,len(data)-1)
            x[0] = data[temp][0]
            x[1] = data[temp][1]
            

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



