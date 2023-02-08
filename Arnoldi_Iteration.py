#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arnoldi_Iteration.py

Code for understanding Arnoldi Iteration; basic, explicit,
and implicitly restarted forms, to solve the eigenvalue problem
for a large array of matrices

@author: Andrew Projansky
last updated: 2/7/2023
"""

import numpy as np
import random
import scipy.linalg as LA
import matplotlib.pyplot as plt
from tqdm import tqdm

def ct(v):
    
    return np.conj(v.T)

def Arnoldi_Base(mat, x, k):
    
    q = [] ; H = np.zeros((k+1,k), dtype='complex')
    q.append(x/LA.norm(x))
    for j in range(k):
        r = mat @ q[j]
        for i in range(j+1):
            H[i,j] = ct(q[i]) @ r ; r = r - H[i,j]*q[i]
        H[j+1,j] = LA.norm(r)
        if H[j+1, j] == 0:
            return q, H[np.arange(0,j,1):]
        q.append(r/H[j+1, j])
    return q, H
#%%
#Base Arnoldi with seperated largest eigenvalue
dim = 20; k = 5
mat = np.random.rand(dim,dim) + 1j*np.random.rand(dim,dim)
x = np.random.rand(dim)
ex_eigs, u = np.linalg.eig(mat)
q, H = Arnoldi_Base(mat, x, k)
Qk = np.array(q)[:k].T ; H = H[:k, :k]
ap_eigs, uprime = np.linalg.eig(H)

plt.plot(np.real(ex_eigs), np.imag(ex_eigs), 'x')
plt.plot(np.real(ap_eigs), np.imag(ap_eigs), 'o', alpha=0.6)
plt.show()
#%%
#Base Arnoldi with clustered largest eigenvalue
dim = 20; k = 5
mat = np.random.rand(dim,dim) - np.random.rand(dim,dim) + 1j * (np.random.rand(dim,dim) - np.random.rand(dim,dim))
x = np.random.rand(dim)
ex_eigs, u = np.linalg.eig(mat)
q, H = Arnoldi_Base(mat, x, k)
Qk = np.array(q)[:k].T ; H = H[:k, :k]
ap_eigs, uprime = np.linalg.eig(H)

plt.plot(np.real(ex_eigs), np.imag(ex_eigs), 'x')
plt.plot(np.real(ap_eigs), np.imag(ap_eigs), 'o', alpha=0.6)
plt.show()
#%%
#Extended Arnoldi
dim = 20; k = 5
mat = np.random.rand(dim,dim) - np.random.rand(dim,dim) + 1j * (np.random.rand(dim,dim) - np.random.rand(dim,dim))
x = np.random.rand(dim)
ex_eigs, u = np.linalg.eig(mat)
q, H = Arnoldi_Base(mat, x, k)