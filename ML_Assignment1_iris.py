# Name: Jazwaur Ankrah

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_file():
    return pd.read_csv('Assignment1\Assignment1\iris .csv').values

def find_distance(point, centroid):
    return np.linalg.norm(point - centroid)

def kmeans(data, k):
    #get random centroids
    np.random.seed(50)
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    
    #get new centroids
    for i in range(5):
        #find closest centroid and assign to cluster
        clusters = [[] for i in range(k)]
        
        for point in data:
            #get distance array and append the smallest value
            distance = [find_distance(point, centroid) for centroid in centroids]
            
            #find closest centroid and add to a cluster
            closest_centroid = np.argmin(distance)
            clusters[closest_centroid].append(point)
        
        #create new centroids based on mean of the cluster points
        new_centroids = np.array([np.mean(cluster, axis=0) if cluster else centroids[i]
                                  for i, cluster in enumerate(clusters)])
        
        #if centroids do not change, break
        if np.all(centroids == new_centroids):
            break

        #update centroids
        centroids = new_centroids

    #return the centroids and clusters
    return centroids, clusters

def iris():
    #import data
    data = load_file()
    data = data[:, 2:4]
    
    #set k value
    k = 3

    #get centroids and clusters
    centroids, clusters = kmeans(data, k)
    
    colors = ['r', 'g', 'b']
    #plot the clusters
    for i, cluster in enumerate(clusters):
        cluster = np.array(cluster)
        plt.scatter(cluster[:, 0], cluster[:, 1], c=[colors[i]],label='Cluster ' + str(i + 1))
    
    #plot the centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], s=100, marker='x', c="black", label='Centroids')
    plt.title('Iris Dataset')
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.legend()
    plt.show()       

#Main
iris()
