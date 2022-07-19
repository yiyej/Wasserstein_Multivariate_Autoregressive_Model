#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 17:32:28 2022

@author: yjiang
"""

########################
# experiment settings: #
########################
import sys

# hyperarameters read from the command line inputs
alpha = float(sys.argv[1]) # bigger alpha, smaller 2-norm of A
N = int(sys.argv[2])    # size of graph
freq = int(sys.argv[3])  # frequence in T of estimation 
isim = int(sys.argv[4])  # isim-th independent simulation group (<<nsim>> simulations) run in parallel 
                         # loop over <<isim>> in task files
T = int(sys.argv[5]) # precision to stop the PGD
#tau = float(sys.argv[6]) # precision to stop the accelerated PGD
tau = 0.0001
nsim = 1 # nb of independent simulations in one simulation group

# all the independent simulations with the same N, alpha values are identical in 
#A values, population Fréchet means, and the noise distribution


##############
# experiment #
##############


#save_path = 'results/alpha0_'+str(alpha)[2:]+'/N'+str(N)+'/tau0_'+str(tau)[2:]+'/'
save_path = 'results/alpha0_'+str(alpha)[2:]+'/N'+str(N)+'/'

import numpy as np
from numpy import linalg as lg
from scipy.interpolate import CubicSpline
from sklearn.isotonic import IsotonicRegression
import time
import pickle 


## noise function: constraint: L-Lipschitz on [0, 1]
gx = [0, 0.2, 0.6, 1]
gy = [0, 0.1, 0.2, 1]
g = CubicSpline(gx, gy)
hi = CubicSpline(0.5*g(np.arange(0,1.001,0.001)) + 0.5*np.arange(0,1.001,0.001), \
                 np.arange(0,1.001,0.001))
    
def epsilon(x,xi): 
    y = hi(x)
    return 0.5*(1+xi)*g(y) + 0.5*(1-xi)*y 

L = 2
# generate A
#nnz = int(N * 0.2) # nb of non zero entries per row 
#A = np.zeros(shape = (N,N))
#for i in range(N):
    #loc_nz = np.random.choice(N, nnz, replace=False)
    #aN1 = np.random.uniform(1, 2, size = nnz)
    #A[i,loc_nz] = aN1/aN1.sum() #condition simplex
# c = lg.norm(A,2)
# A /= (L+alpha)*c #condition L2 norm 

with open('results/alpha'+str(alpha)+'_N'+str(N)+'_A','rb') as fp:
    A = pickle.load(fp)
        

for nn in range(nsim):
    print(nn)
    # synthesize centered data tilde_F^-1_i,t from model (3.5):
    p_list = np.arange(0,1,0.01) # granularity applied in the numeric methods to approaximate the function values
    qt_ct = np.zeros(shape=(N,T+1,len(p_list)))
    for n in range(N):
        qt_ct[n,0,:] = p_list
    for t in range(1,T+1):
        for i in range(N):
            pred = A[i,:].dot(qt_ct[:,t-1,:]-p_list) + p_list
            xi = np.random.uniform(-1,1)
            qt_ct[i,t,:] = epsilon(pred,xi)
    
    # synthesize raw data F^-1_i,t:
    # generate and multiply "back" the frechet means F^-1_i,Oplus
    #   constraint for the population frechet means F^-1_i,Oplus: continuous (Assumption 2)   
    gx = [0, 0.2, 0.6, 1]
    gy = [0, 0.1, 0.2, 1]
    g = CubicSpline(gx, gy)
    
    qt = qt_ct.copy()
    for i in range(N):
        gy = [0, 0.1, 0.2 + 0.2*i/N, 1]
        fmean_popu = CubicSpline(gx, gy)
        for t in range(T+1):
            f_it = CubicSpline(p_list,qt_ct[i,t,:])
            qt[i,t,:] = f_it(fmean_popu(p_list))
    
    # accelerated projected gradeint descent    
    def Proj_odr(y):
        x = IsotonicRegression().fit_transform(np.arange(N), y.flatten())
        return np.clip(x, 0, 1)
    
    def mulN(y):
        #return the multiplication yN
        y_ = y[::-1] #flip vector y
        y_ = np.hstack((y_[0], np.diff(y_)))
        return y_[::-1]
    
    err = []
    spd = []
    rtime = []
    for Tt in range(2, T+1, freq):
        #print(Tt)  
        qt_ct_emp = qt[:,:Tt,:].copy()
               
        start_ac = time.time()
        #center
        for i in range(N):
            fmean_emp = qt_ct_emp[i,1:,:].mean(axis = 0)
            for t in range(Tt):
                f_it = CubicSpline(p_list,qt_ct_emp[i,t,:])
                fmean_emp_inv = CubicSpline(fmean_emp, p_list)
                qt_ct_emp[i,t,:] = f_it(fmean_emp_inv(p_list))  
        
        Gamma0 = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                m = 0
                for t in range(Tt-1):
                    m += np.trapz((qt_ct_emp[i,t,:]-p_list)*(qt_ct_emp[j,t,:]-p_list), x=p_list)
                Gamma0[i,j] = m/(Tt-1)
    
        Gamma1 = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                m = 0
                for t in range(Tt-1):
                    m += np.trapz((qt_ct_emp[i,t+1,:]-p_list)*(qt_ct_emp[j,t,:]-p_list), x=p_list)
                Gamma1[i,j] = m/(Tt-1) 
    
        beta = 0.5
        eta = np.ones(N)
        iter_max = 500000
        Z = np.zeros((N,N))
        Z_old = np.zeros((N,N))
        Z_tilde = np.zeros((N,N))           
        for i in range(N):
            err_ = []
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
                err_.append(lg.norm(Z_tilde[i,:] - Z_old[i,:]))
                if err_[-1] < tau: break
                Z[i,:] = Z_tilde[i,:].copy()
        A_hat = np.hstack((Z[:,0].reshape(-1,1), np.diff(Z)))
        end_ac = time.time()   
        rtime.append(end_ac-start_ac)
        err.append(lg.norm(A_hat - A))
        spd.append(1/np.sqrt(Tt-1))    
    
    # Save the results 
    with open(save_path+str(isim*nsim+nn),'wb') as fp:
        pickle.dump((err, spd, rtime),fp)



