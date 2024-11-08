# Name: Jazwaur Ankrah

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_file():
    #import file
    df = pd.read_csv('kmtest.csv', header=None).values
    return df

def normalize_data(input_data):
    #use equation (data - data.min) / (data.max - data.min)
    min = input_data.min(axis=0)
    max = input_data.max(axis=0)
    data = (input_data - min) / (max - min)
    return data

def find_distance(point1, point2):
    #find distance between point and centroid
    return np.linalg.norm(point1 - point2)

def kmeans(data, k):
    #produce random centroids
    np.random.seed(50)
    centroids = data[np.random.choice(data.shape[0], k, replace = False)]
    
    #get new centroids
    for i in range(50):
        #create array of clusters for each centroid based on k value
        clusters = [[] for x in range(k)]
        
        #find closest centroid, assign to cluster, and create new centroids
        for point in data:
            #create array of distance of all centroids
            distance = [find_distance(point, centroid) for centroid in centroids]
            
            #find closest centroid and add to a cluster
            closest_centroid = np.argmin(distance)
            clusters[closest_centroid].append(point)

        #create new centroids based on mean of the cluster points
        new_centroids = np.array([np.mean(cluster, axis=0) if cluster else centroids[i]
                                      for i, cluster in enumerate(clusters)])
            
        #check to see if centroids changed values
        if np.all(centroids==new_centroids):
                break

        #if not, update the centroid array
        centroids = new_centroids

    #return the centroids and clusters
    return centroids, clusters

def Non_normalized():
    data = load_file()
    
    #make array of k values
    k_values = [2, 3, 4, 5]
    
    #create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    #create color array
    colors = ['r', 'g', 'b', 'c', 'm']
    
    #call all kmeans
    for idx, k in enumerate(k_values):
        centroids, clusters = kmeans(data, k)
        ax = axes[idx]
       
        #plot clusters
        for i, cluster in enumerate(clusters):
            cluster = np.array(cluster)
            if len(cluster) > 0:
                ax.scatter(cluster[:, 0], cluster[:, 1],color=colors[i%len(colors)],label=f'Cluster {i+1}')
       
        #plot centroids
        ax.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='x', s=100, label='Centriods')
       
        #display plot
        ax.set_title(f'Non_normalized K-Means Clustering (k={k})')
        ax.grid(False)
        ax.set_xlim(0, 17)
        ax.set_ylim(0, 11)
        ax.set_aspect('equal')

    #display plot
    plt.show()

def Normalized():
    reg_data = load_file()
    data = normalize_data(reg_data)
   
    #make array of k values
    k_values = [2, 3, 4, 5]
   
    #create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(10,10))
    axes = axes.flatten()
    
    #create color array
    colors = ['r', 'g', 'b', 'c', 'm']
    
    #call all kmeans
    for idx, k in enumerate(k_values):
        centroids, clusters = kmeans(data, k)
        ax = axes[idx]
       
        #plot clusters
        for i, cluster in enumerate(clusters):
            cluster = np.array(cluster)
            if len(cluster) > 0:
                ax.scatter(cluster[:, 0], cluster[:, 1], color=colors[i%len(colors)], label=f'Cluster {i+1}' )
       
        #plot centroids
        ax.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='x', s=100, label='Centriods')
        
        ax.set_title(f'Normalized K-Means Clustering (k={k})')
        ax.grid(False)
        ax.set_xlim(-.2, 1.2)
        ax.set_ylim(-.2, 1.2)
        ax.set_aspect('equal')
   
    #display plot
    plt.show()

#Main
Non_normalized()
Normalized()



