# Will eventually be a series of test that compare the running time and performance of my 
# kMeans vs sci-kit learn kMeans.  Doesn't save results, just runs algorithm.

import time
import kMeans as km
# reload(kMeans)  # Call if any changes have been made to kMeans.py
import sklearn.cluster as sklearn

# Create data
dataSet = km.createDataSet(10,50,5,10,1)

# Cluster numbers to test
Nclusters = 12  # Slightly larger than true cluster number
Num_iters = 50

# Test my implementation

start_time = time.clock()
for i in range(Num_iters):
	final_cluster_pos, cost = km.kMeans(dataSet, Nclusters)
        
avg_run_time = (time.clock() - start_time)/Num_iters
print "Average run time per my kMeans iteration is", avg_run_time, "s" 

# Test scikit-learn kMeans

start_time_scikit = time.clock()

skm = sklearn.KMeans(init='random', n_clusters=Nclusters, n_init=Num_iters)
skm.fit(dataSet)

avg_run_time_scikit = (time.clock() - start_time_scikit)/Num_iters

print "Average run time per sklearn kMeans iteration is", avg_run_time_scikit, "s" 

print "On my computer the sklearn implemention runs", avg_run_time/avg_run_time_scikit, "times faster"
