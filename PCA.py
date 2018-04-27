# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 22:37:24 2017

@author: LATA
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io
from numpy import linalg as LA

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
final_data.shape
eigenVectors.shape
pcaScores = np.matmul(final_data, eigenVectors)

pcaResults = {'data': final_data,
              'mean_centeredData':finalData_meanCentered,
              'PC_variance': eigenValues,
              'loadings': eigenVectors,
              'scores': pcaScores}

transformed_data=pcaResults['scores'][:,0:2]

transformed_data.real
transformed_data.shape

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_title('Reconstructed data')
ax.scatter(transformed_data[:, 0], transformed_data[:, 1], color='blue')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
fig.show()
