#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tensor Networks Experiments/Introductions 
Created on Sun Oct 30 20:27:10 2022

@author: andrewprojansky
"""

import numpy as np

class TensorNetwork:
    
    def __init__(
        self, 
        dim, 
        state
        
        
    ):
        
        self.dim = dim
        self.state = state
        self.TNW = {}
        
    def Psi_1(self, mdim):
        '''
        Initializes first Psi matrix to be 2 x d^(L-1)
        '''
        
        Psi = np.zeros((2,mdim//2))
        for i in range(mdim//2):
            Psi[0,i] = self.state[i]
            Psi[1,i] = self.state[mdim//2+i]
        return Psi
    
    def Make_TN(self):
        '''
        Makes TN for each site, getting A-sigma matrices at each site with left 
        normalization at all except for last
        '''
        
        Psi = self.Psi_1(self.dim**2)
        prank = 1
        for i in range(dim-1):
            u, s, vh = np.linalg.svd(Psi, full_matrices=True)
            s = np.around(s, decimals = 10)
            s = s[0:len(np.nonzero(s)[0])]
            rank = len(s)
            u = u[:, :rank]
            u = u.reshape((2,prank,rank))
            self.TNW[i] = u
            vh = vh[:rank,:]
            Psi = np.reshape(np.matmul(np.diag(s), vh), (rank*2, 2**(dim-2-i)))
            print(Psi)
            prank = rank
        Psi = Psi.reshape(2, 1, rank)
        self.TNW[dim-1] = Psi
    
    def Multiplication(TN, bi: str):
        if len(TN.keys()) != len(bi):
            print('Need new string')
            return
        else:
            return np.linalg.multi_dot([x[int(bi[i])] for i,x in enumerate(TN.values())])
    
dim = 4
initial_state = np.array([1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1])
TN = TensorNetwork(initial_state, dim)
TN.Make_TN()
TN.Multiplication(TN, '1111')
