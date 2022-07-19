#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 19:05:31 2022

@author: yiye
"""
import requests
import pandas as pd
import numpy as np
import numpy.linalg as lg
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.isotonic import IsotonicRegression
import time

EU_sch_UK = ['Austria','Belgium','Bulgaria','Croatia','Cyprus','Czechia',\
        'Denmark','Estonia','Finland','France','Germany','Greece','Hungary',\
        'Iceland', 'Ireland', 'Italy','Latvia', 'Liechtenstein', 'Lithuania',\
        'Luxembourg','Malta', 'Monaco', 'Netherlands', 'Norway', 'Poland',\
        'Portugal','Romania', 'San Marino', 'Slovakia','Slovenia','Spain',\
        'Sweden', 'Switzerland', 'United Kingdom']


# download the age population data 
N = len(EU_sch_UK)
T = 40
x_list = np.arange(0,101)
X_pop = np.zeros(shape=(N,T,len(x_list)),dtype=np.int64)
for i,name in enumerate(EU_sch_UK):
    print(i)
    for t,yr in enumerate(np.arange(1996, 2036)):
        data_url = 'https://api.census.gov/data/timeseries/idb/1year?get=FIPS,'\
            +'POP,AGE&YR={}&SEX=0&NAME={}&key=fd013bcf9af322afbc38abb2b973e89f3638b0ab'.format(yr,name)
        response=requests.get(data_url)
        data=response.json()
        df=pd.DataFrame(data[1:], columns=data[0])
        X_pop[i,t,:]=df['POP'].to_numpy(dtype=np.int64)
        
        

# population counts (a collection of mass on points), which is not histogram. 
# we do not visualize the data by the strict histogram, 
# because we do not suppose the true distribution has density.
plt.figure()
plt.bar(x_list, X_pop[i,t,:]) 

def generate_qt_fun(x):
    """
    Retrieve the quantile function from the population counts through ages 1, 2, ..., 99, 100+.
    We consider all the age 100 plus as age 100. Thus the domain of the observations is [0,100].
    We then scale the domain by 100 to have the support [0,1], required by the proposed model. 
        input: x, is a vector of the population counts of length 101.
        output: qt_fun, quntile function which is 
                        1. continuous (Assumption 2), 
                        2. strictly increasing between [max(qt_fun(0)), 1], (for the simplicity to code the inverse)
                        3. qt_fun(1) = 1 (Assumption 2).
    """
    # we first approximate ecdf as a strictly increasing* continuous function 
    # by interpolating the dicrete evaluated points. 
    # Then we approximate the quantilfe as the inverse of the ecdf approximated.
    # *to this end, we evaluate the ecdf on the points 0, 1/100, 2/100, ..., 1, based on 
    # the population counts. Since the population counts does not contain zero,
    # the ecdf evaluated will increase definitely from one age point to another. 
    # Thus the interpolant will alwalys be strictly incresing so is its inverse.  
    dist_x = np.arange(0,101)/100
    ecdf = np.cumsum(x)/np.sum(x)
    qt_fun = interp1d(np.hstack((0,ecdf)), np.hstack((0,dist_x)))
    return qt_fun


p_list = np.linspace(0,1,501) # granularity applied in the numeric methods to approaximate the function values
qt = np.zeros((N, T, len(p_list)))
for i in range(N):
    for t in range(T):
        qt_fun = generate_qt_fun(X_pop[i,t,:])
        qt[i,t,:] = qt_fun(p_list)

    
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
        qt_fun = generate_qt_fun(X_pop[i,t,:])
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
    
#np.savetxt("A_age_pop.csv", A, delimiter=",")






