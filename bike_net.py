#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 22:21:35 2021

@author: yiye
"""
import pandas as pd
import numpy as np
import numpy.linalg as lg
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.isotonic import IsotonicRegression
import time

from scipy.interpolate import interp1d
from statsmodels.distributions.empirical_distribution import ECDF

X_hour = np.copy(pd.read_csv("X_hour_paris.csv"))[:,1:]

T, N = X_hour.shape
X_jump= np.zeros((N, 24, T//24))
for t in range(24):
    for i in range(N): 
        ind = np.arange(t, 24*(T//24), 24)
        X_jump[i,t,:] = X_hour[ind,i].copy()
        X_jump[i,t,X_jump[i,t,:] > 1] = 1
        

def generate_qt_fun(x):
    """
        input: x, is a vector of the iid observations.
        output: qt_fun, quntile function which is 
                        1. continuous (Assumption 2), 
                        2. strictly increasing between [max(qt_fun(0)), 1], (for the simplicity to code the inverse)
                        3. qt_fun(1) = 1 (Assumption 2).
    """
    dist_x = np.unique(x)
    if dist_x[-1] < 1: 
        ecdf = ECDF(np.hstack((x,1)))
        dist_x = np.hstack((dist_x,1))
    else:
        ecdf = ECDF(x)
    # we first approximate ecdf as a strictly increasing* continuous function 
    # by interpolating the dicrete evaluated points. 
    # Then we approximate the quantilfe as the inverse of the ecdf approximated.
    # *to this end, we do not evaluate the ecdf on a grill on x
    # since we will likely have ties, especially when the sample size is limited
    # instead, we choose to evaluate on the unique data points, since the ecdf will increase 
    # definitely from one data point to another. 
    # this method to retrieve quantile from data will awlays give 
    # a strictly increasing continuous function even with small sample size  
    qt_fun = interp1d(np.hstack((0,ecdf(dist_x))), np.hstack((0,dist_x)))
    return qt_fun
       
        
N, T, _ = X_jump.shape
p_list = np.linspace(0,1,501) # granularity applied in the numeric methods to approaximate the function values
qt = np.zeros((N, T, len(p_list)))
for i in range(N):
    for t in range(T):
        qt_fun = generate_qt_fun(X_jump[i,t,:])
        qt[i,t,:] = qt_fun(p_list)
            
           
plt.figure()
for t in range(20):
    plt.plot(p_list, qt[5,t,:], color = (0.1, 0.2, 0.05*t))
    plt.pause(0.5) 

    
#center
start_ac = time.time()
qt_mean = np.mean(qt, axis = 1)
        
qt_ct = np.zeros((N, T, len(p_list)))
for i in range(N):
    n0 = sum(qt_mean[i,:] == 0)
    dist_x = qt_mean[i,(n0-1):]
    dist_y = p_list[(n0-1):]
    dist_fun = interp1d(dist_x, dist_y)
    for t in range(T):       
        qt_fun = generate_qt_fun(X_jump[i,t,:])
        qt_ct[i,t,:] = qt_fun(dist_fun(p_list)) - p_list

        
#construct Gamma(0)
Gamma0 = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        m = 0
        for t in range(T-1):
            m += np.trapz(qt_ct[i,t,:]*qt_ct[j,t,:], x=p_list)
        Gamma0[i,j] = m/(T-1)
        
#construct Gamma(1)
Gamma1 = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        m = 0
        for t in range(T-1):
            m += np.trapz(qt_ct[i,t+1,:]*qt_ct[j,t,:], x=p_list)
        Gamma1[i,j] = m/(T-1)  
     

# plt.figure()
# sns.heatmap(Gamma0)


# accelerated projected gradeint descent    
def Proj_odr(y):
    x = IsotonicRegression().fit_transform(np.arange(N), y.flatten())
    return np.clip(x, 0, 1)

def mulN(y):
    #return the multiplication yN
    y_ = y[::-1] #flip vector y
    y_ = np.hstack((y_[0], np.diff(y_)))
    return y_[::-1]
    
beta = 0.5
eta = np.ones(N)
iter_max = 20000
Z = np.zeros((N,N))
Z_old = np.zeros((N,N))
Z_tilde = np.zeros((N,N))

for i in range(N):
    print(i)
    err = []
    def f_i(x):
        x_ = np.hstack((x[0], np.diff(x)))
        val = x_.dot(Gamma0).dot(x_.reshape(-1,1)) \
                -2*Gamma1[i,:].dot(x_.T)
        return val
    for k in range(iter_max):
        if k == (iter_max-1): print("iter_max reached.")
        omega = k/(k+3)
        y = Z[i,:] + omega*(Z[i,:] - Z_old[i,:])
        y_ = np.hstack((y[0], np.diff(y)))
        grad = mulN(y_.dot(Gamma0) - Gamma1[i,:]) 
        while True:
            Z_tilde[i,:] = Proj_odr(y - eta[i]*grad)
            issmaller = f_i(Z_tilde[i,:]) - f_i(y) \
                        -(Z_tilde[i,:] - y).dot(grad.reshape(-1,1)) \
                        - lg.norm(Z_tilde[i,:] - y)**2/2/eta[i] <=0
            if issmaller: break
            eta[i] *= beta
        Z_old[i,:] = Z[i,:].copy()
        err.append(lg.norm(Z_tilde[i,:] - Z_old[i,:]))
        if err[-1] < 0.0001: break
        Z[i,:] = Z_tilde[i,:].copy()
A = np.hstack((Z[:,0].reshape(-1,1), np.diff(Z)))
end_ac = time.time()        
  

plt.figure()
sns.heatmap(A)

    
#np.savetxt("A.csv", A, delimiter=",")




