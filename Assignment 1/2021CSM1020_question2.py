# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 21:32:00 2022

@author: Ananya_Chaudhuri
"""
import matplotlib.pyplot as plt
import numpy as np

N = 20

X = np.linspace(0, 2*np.pi, num=N)
Y = np.cos(2*np.pi*X)+X/(2*np.pi)+np.random.normal(0,0.004,N) #Add noise of mean 0 and standard deviation 1 

# Plot
plt.figure
plt.plot(X, Y, 'bo')
plt.title("Noisy Cosine data points")
plt.show()
np.random.seed(1)
RMSE_Error=[]
plt.figure(figsize=(12,12))
for j,i in enumerate([1,2,3,5,7,10]):
    z = np.polyfit(X, Y, i) # Fitting the polynomial and finding coefficients
    p = np.poly1d(z)# zth order polynomial construction
    y = p(X)
    MSE = np.square(np.subtract(Y,y)).mean()
    RMSE = np.sqrt(MSE)
    RMSE_Error.append(RMSE)
    #print("Root Mean Square Error for M value ",i,": ",RMSE)
#     plt.figure
    plt.subplot(3, 2,j+1)
    plt.scatter(X,y, color='red')
    plt.plot(X, Y, 'g-')
    plt.plot(X, Y, 'bo', X,y,'r-')
    plt.legend(('Training data', 'Fitting with M='+str(i)))
# =============================================================================
# import matplotlib.pyplot as pl
# pl.figure
# pl.scatter([1,2,3,5,7,10],RMSE_Error)
# pl.xlabel("Different values of M")
# pl.ylabel("RMSE")
# pl.show()
# =============================================================================
sample_points = 200

x_sampled = np.linspace(0, 2*np.pi, num=sample_points)
t_sampled = np.cos(2*np.pi*x_sampled)+x_sampled/(2*np.pi)+np.random.normal(0,0.004,sample_points) #Add noise of mean 0 and standard deviation 1 

# Plot
# =============================================================================
# plt.figure
# plt.plot(x_sampled, t_sampled, 'bo')
# plt.title("Noisy Cosine data points")
# plt.show()
# =============================================================================
np.random.seed(1)
RMSE_Error=[]
plt.figure(figsize=(12,12))
for j,i in enumerate([1,2,3,5,7,10]):
    z = np.polyfit(x_sampled, t_sampled, i) # Fitting the polynomial and finding coefficients
    p = np.poly1d(z)# zth order polynomial construction
    y = p(x_sampled)
    MSE = np.square(np.subtract(t_sampled,y)).mean()
    RMSE = np.sqrt(MSE)
    RMSE_Error.append(RMSE)
    #print("Root Mean Square Error for M value ",i,": ",RMSE)
#     plt.figure
    plt.subplot(3, 2,j+1)
    plt.scatter(x_sampled,y, color='red')
    plt.plot(x_sampled, t_sampled, 'g-')
    plt.plot(x_sampled, t_sampled, 'bo', x_sampled,y,'r-')
    plt.legend(('Training data', 'Fitting with M='+str(i)))