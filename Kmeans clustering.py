# -*- coding: utf-8 -*-
"""

@author: LATA
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import io
from numpy import linalg as LA

np.set_printoptions(threshold=1000)
def euclediandist(x,y):
    distance = 0.0
    for i in range(0,len(x)):
        distance=distance+ (x[i] - y[i])**2
    return math.sqrt(distance)

def kMeans(data, K, maxIteration = 110, progress = None):
    
    #Selection of centroids
    centroids = data[np.random.choice(np.arange(len(data)), K), :]
    for i in range(maxIteration):
        # Assignment step
        Cluster= np.array([np.argmin([euclediandist(y,x) for y in centroids]) for x in data])
        # Update step
        centroids= [data[Cluster == k].mean(axis = 0) for k in range(K)]
        if progress != None: progress(data, Cluster, np.array(centroids))
    return np.array(centroids) , Cluster

def plot(X, Cluster, centroids, keep = False):
   
    plt.cla()
    plt.plot(X[Cluster == 0, 0], X[Cluster == 0, 1], '.r',
         X[Cluster == 1, 0], X[Cluster == 1, 1], '.b',
         X[Cluster == 2, 0], X[Cluster == 2, 1], '.g')
    plt.plot(centroids[:,0],centroids[:,1],'+m',markersize=20)
    plt.draw()
    if keep :
        plt.ioff()
        plt.show()

data=pd.read_csv("E:/Lata/UNCC/Fall 17/ML/SCLC_study_output_filtered.csv")

data1= data.to_csv(header=None, index = False)
data2= pd.read_csv(io.StringIO(u""+data.to_csv(header=None,index=False)), header=None)
final_data= data2.drop(data2.columns[0], axis=1)

columnMean = final_data.mean(axis=0)
columnMeanAll = np.tile(columnMean, reps=(final_data.shape[0], 1))
finalData_meanCentered = final_data - columnMeanAll

covarianceMatrix = np.cov(final_data.astype(float),rowvar=False)
print('Covariance Matrix:\n', covarianceMatrix)

# calculating eigen values and eigen vectors
eigenValues, eigenVectors = LA.eig(covarianceMatrix)
II = eigenValues.argsort()[::-1]
eigenValues = eigenValues[II]
eigenVectors = eigenVectors[:, II]
eigenValues=eigenValues

pcaScores = np.matmul(final_data, eigenVectors)

pcaResults = {'data': final_data,
              'mean_centeredData':finalData_meanCentered,
              'PC_variance': eigenValues,
              'loadings': eigenVectors,
              'scores': pcaScores}

transformed_data=pcaResults['scores'][:,0:2]

#Calling the kMeans function, K is number of clusters
centroids, Cluster = kMeans(transformed_data, K = 2, progress = plot)

print(Cluster)

#Plotting the centroids and clustered data
plot(transformed_data, Cluster, centroids, True)