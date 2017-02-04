# http://stanford.edu/~cpiech/cs221/handouts/kmeans.html
import numpy as np
import math

# K-Menas:
# input: dataset, number of cluster
# output: centroids

def stop_iteration(centroids, old_centroids, m):
    
    if (centroids == old_centroids) | (m < M):
        return True
    
    return False

def assign_labels(data_set, current_centroids):
    total_row = data_set.shape[0]
    num_centroids = current_centroids.shape[0];
    labels = []
    for i in range(total_row):
        point_vector = data_set[i]
        current_min = 2**31
        which_center = -1
        for j in range(num_centroids):
            centorid_vector = current_centroids[j]
            current_distance = 1.0 * math.sqrt(sum((point_vector - centorid_vector)**2))
            if (current_distance  < current_min):
                current_min = current_distance
                which_center = j

        labels += [which_center]
    
    return np.array(labels)
                

def getCentroids(data_set, labels_array, k):
    assert data_set.shape[0] == len(labels_array)
    centroids = []
    
    for i in range(k):
        mask = (labels_array == i)
        data_set_i = data_set[mask]
        center_i = data_set_i.mean(axis = 0)
        centroids += [list(center_i)]
        
    return np.array(centroids)


def k_means(data_set, k):

    # datat_set: 2-d array, (x, y): x-rows of data, y features
    # randomly choose k data points from data_set.
    total_row = data_set.shape[0]

    # randomly choose k points from [0, total_row)
    chooise_k_indexs = np.random.randint(0, total_row, k)
    current_centroids = data_set[chooise_k_indexs]
    old_centroids = None
    m = 0
    while !(stop_iteration(current_centroids, old_centroids, m)):
        m += 1

        # based on the current_centroids, assign labels to all the data
        # labels_array: 1-d array
        labels_array = assign_labels(data_set, current_centroids)
        old_centroids = current_centroids
        current_centroids = getCentroids(data_set, labels_array, k)

    return current_centroids
