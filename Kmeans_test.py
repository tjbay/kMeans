# Finds the average kMeans performance as a function of the number of clusters
# Produces the "elbow plot".  Also calculate the average time/iteration of the 
# entire algorithm.

import matplotlib.pyplot as plt
import time
import kMeans as km
# reload(kMeans)  # Call if any changes have been made to kMeans.py

# Create data
dataSet = km.createDataSet(10,50,5,10,1)

# Cluster numbers to test
Nrange = range(5,15)
average_num = 2

average_fit = [0 for i in Nrange]

start_time = time.clock()

for index, NumC in enumerate(Nrange):
    print 'Testing Number of Clusters =', NumC
    fit_sum = 0
    for i in range(average_num):
        final_cluster_pos, cost = km.kMeans(dataSet, NumC)
        fit_sum += cost
    average_fit[index] = fit_sum/average_num
        
run_time = time.clock() - start_time

Niterations = len(Nrange)*average_num

print "Average run time per kMeans iteration is", run_time/Niterations, "s" 

plt.plot(Nrange, average_fit)