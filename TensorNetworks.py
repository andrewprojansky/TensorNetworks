#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tensor Networks Experiments/Introductions 
Created on Sun Oct 30 20:27:10 2022

@author: andrewprojansky
"""

import numpy as np

def Psi_1(i_s, dim):
    
    Psi = np.zeros((2,dim//2))
    for i in range(dim//2):
        Psi[0,i] = i_s[i]
        Psi[1,i] = i_s[dim//2+i]
    return Psi

def Make_TN(Psi, dim):
    
    TN = {}
    prank = 1
    for i in range(dim-1):
        u, s, vh = np.linalg.svd(Psi, full_matrices=True)
        s = np.around(s, decimals = 10)
        s = s[0:len(np.nonzero(s)[0])]
        rank = len(s)
        u = u[:, :rank]
        u = u.reshape((2,prank,rank))
        TN[i] = u
        vh = vh[:,:rank]
        Psi = np.reshape(np.matmul(np.diag(s), np.transpose(vh)), (rank*2, 2**(dim-2-i)))
        prank = rank
    Psi = Psi.reshape(2, 1, rank)
    TN[dim-1] = Psi
    return TN

def Multiplication(TN, bi: str):
    if len(TN.keys()) != len(bi):
        print('projansky youre stinky')
        return
    else:
        return np.linalg.multi_dot([x[int(bi[i])] for i,x in enumerate(TN.values())])
    

dim = 4
initial_state = np.array([1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1])
Psi = Psi_1(initial_state, 2**dim)
TN = Make_TN(Psi, dim)
print(TN)
