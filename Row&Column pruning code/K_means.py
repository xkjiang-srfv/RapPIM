
import random
import pandas as pd
import numpy as np
import copy
import math


def Dis(dataSet, centroids, k):
    if len(centroids) < k:
        centroids = np.append(centroids, random.sample(list(dataSet), k-len(centroids)), axis=0)
    
    clalist=[]
    for data in dataSet:
        diff = np.tile(data, (k, 1)) 
        mul_Diff = np.multiply(diff, centroids)
        mul_Dist = np.sum(mul_Diff, axis=1)   
        clalist.append(mul_Dist) 
    clalist = np.array(clalist) 
    return clalist 


def classify(dataSet, centroids, k):
    clalist = Dis(dataSet, centroids, k)
    minDistIndices = np.argmax(clalist, axis=1)    
    newCentroids = pd.DataFrame(dataSet).groupby(minDistIndices).mean() 
    newCentroids = newCentroids.values
    for centro in newCentroids:
        sorted_data=np.argsort(centro)  
        value = 1
        for valueIndex in sorted_data:
            centro[valueIndex] = value
            value += 1
    
    if len(newCentroids) != len(centroids):
        changed = 1  
    else:
        changed = newCentroids - centroids 

    return changed, newCentroids


def euler_distance(point1: list, point2: list) -> float:
    
    distance = 0.0
    for a, b in zip(point1, point2):
        distance += a*b
    return distance
    

def get_closest_dist(point, centroids):
    min_dist = math.inf 
    for i, centroid in enumerate(centroids):
        dist = euler_distance(centroid, point)
        if dist < min_dist:
            min_dist = dist
    return min_dist


def kpp_centers(data_set: list, k: int) -> list:
    cluster_centers = []
    cluster_centers.append(random.choice(data_set))
    d = [0 for _ in range(len(data_set))]
    for _ in range(1, k):
        total = 0.0
        for i, point in enumerate(data_set):
            d[i] = get_closest_dist(point, cluster_centers) 
            total += d[i]
        total *= random.random()
        for i, di in enumerate(d):
            total -= di
            if total > 0:
                continue
            cluster_centers.append(data_set[i])
            break
    return cluster_centers


def kmeans(dataSet, k):
    valueSet = np.zeros(dataSet.shape, dtype=int)  
    for index in range(len(dataSet)):
        data = dataSet[index]
        value = valueSet[index]
        sorted_data=list(map(abs,data))  
        sorted_data=np.argsort(sorted_data)  
        i = 1  
        for valueIndex in sorted_data:
            value[valueIndex] = i
            i += 1

    centroids=kpp_centers(valueSet, k)
    
    i=100
    changed, newCentroids = classify(valueSet, centroids, k)
    while np.any(changed != 0) and i > 0:
        changed, newCentroids = classify(valueSet, newCentroids, k)
        i=i-1
        print("第{}次迭代".format(100-i))
 
    centroids = sorted(newCentroids.tolist())  
 
    clalist = Dis(valueSet, centroids, k) 
    minDistIndices = np.argmax(clalist, axis=1)  
    return minDistIndices


def getCluster(input, clusters_num):
    if len(input.shape) == 2:  
        fcValues = input.detach().cpu().numpy()  
        clusterIndex = kmeans(fcValues, clusters_num)
    elif len(input.shape) == 4:  
        kernel_size = input.shape[3]  
        preShape = input.shape[:2]  
        inputCut = input.view(preShape[0]*preShape[1], kernel_size*kernel_size)  
        convValues = inputCut.detach().cpu().numpy() 
        clusterIndex = kmeans(convValues, clusters_num)  
        clusterIndex.resize(preShape)
    else:
        clusterIndex = None
    
    return clusterIndex