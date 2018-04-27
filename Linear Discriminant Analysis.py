# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 20:47:32 2017

@author: Lata
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import linalg as LA

dataIn=pd.read_csv("E:/Lata/UNCC/Fall 17/ML/Final exam/dataset_1.csv")
v1=dataIn["V1"]
v2=dataIn["V2"]
data=np.array([v1,v2])

#plot V2 vs V1
plt.plot(v2,v1,'go')

mean_data=data.mean()
dataForPCA_meanCentered = data - mean_data
covariance_matrix=np.cov(dataForPCA_meanCentered)

eigenValues, eigenVectors = LA.eig(covariance_matrix)
II = eigenValues.argsort()[::-1]
eigenValues = eigenValues[II]
eigenVectors = eigenVectors[:, II]
eigenValues=eigenValues

x_pc1 = [0,-50*eigenVectors[0,0:1]]
y_pc1= [0,-50*eigenVectors[1,0:1]]

#plotting raw data and PC1 axis
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_title('raw data and PC axis')  
ax.plot(v1,v2,'bo' ,x_pc1,y_pc1,'g-')
fig.show()

#LDA
data_for_LDA=data.T
V1=np.asarray(data_for_LDA[0:30,:])
V2=np.asarray(data_for_LDA[30:60,:])

mean1=np.mean(V1,axis=0).reshape(2,1)
mean2=np.mean(V2,axis=0).reshape(2,1)

a1=(np.transpose(V1)-mean1)
a2=(np.transpose(V2)-mean2)

S1=np.dot(a1,a1.T)
S2=np.dot(a2,a2.T)

s_within=S1+S2
#calculating W
W=np.dot(LA.inv(s_within),(mean1-mean2))

y=np.dot(W.T,data_for_LDA.T)
y_i=np.zeros(30).reshape(1,30)

#Projection of raw data on W
fig=plt.figure()
ax=fig.add_subplot(1, 1, 1)
ax.plot(y[0,0:30],y_i.T,'ro',y[0,30:60],y_i.T,'go')
fig.show()
print("There is a clear separation of data when projected on W")

x_W = [0,-50*W[0,0:1]]
y_W= [0,-50*W[1,0:1]]

#Add W axis to plot, plot contains raw data, pc axis and W axis
plt.plot(v1,v2,'bo',x_pc1,y_pc1,'g-',x_W,y_W,'r-')

#calculate PC variance for PC1 and PC2
PC_variance1= eigenValues[0] / sum(eigenValues)
PC_variance2= eigenValues[1] / sum(eigenValues)


