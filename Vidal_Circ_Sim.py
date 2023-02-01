#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vidal_Circ_Sim.py
---------------------------------------------------------------------
Simulating quantum circuits using tensor networks, following methods
outlines in Vidal [2003]. 

    by Andrew Projansky - last modified February 1st
"""

import numpy as np
from numpy import linalg as LA
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

H = 1/np.sqrt(2) * np.array([[1,1],[1,-1]])
S = np.array([[1,0],[0,1j]])
T = np.array([[1,0],[0,np.exp(1j*np.pi/4)]])
X = np.array([[0,1],[1,0]])
Y = np.array([[0,-1j],[1j,0]])
Z = np.array([[1,0],[0,-1]])
CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])

class Circuit:
    """
    Class for circuits composed of matrix product operators acting on 
    matrix product states
    
    Parameters
    ----------
    N : int
        number of qubits
    max_chi : int
        max bond dimension in MPS
    init_state: string, array
        accepts string arguements 'zero','rand_loc', and 'random' for 
        generating initial states. Also accepts manually written array 
        as initial state
    c_center: int
        canonization center for MPS 
        
    Attributes
    ----------
    Psi: array
        empty array of tensors, to be filled after initializing circuit
    """
    
    def __init__(self, N=10, max_chi = 2**(10//2),
                init_state = 'zero', c_center = 9):
        self.N = N
        self.max_chi = max_chi
        self.init_state = init_state
        self.Psi = [0 for x in range(N)]
        self.c_center = c_center
        
    def init_Circuit(self):
        """
        Initializes circuit by defining Psi dependant on initial state
        input, then canonizes the state while normalizing all tensors

        """
        ### Initialize Psi
        if self.init_state == 'zero':
            for i in range(self.N):
                self.Psi[i] = np.array([[[1],[0]]])
        if self.init_state == 'rand_loc':
            for i in range(self.N):
                a = random.random()
                self.Psi[i] = np.array([[[a],[1-a]]])
        if self.init_state == 'random':
            self.Psi[0] = np.random.rand(1,2,min(self.max_chi, 2))
            for k in range(1,N):
                self.Psi[k] = np.random.rand(self.Psi[k-1].shape[1],2,
                          min(min(self.max_chi,self.Psi[k-1].shape[2]*2),
                              2**(N-k-1)))
        if type(self.init_state) == type(np.array([])):
            self.Psi = self.init_state
           
        ### Canonize Psi
        self.canonize_psi()
            
    def left_canonize(self, site):
        """
        Left canonoize sites by performing SVD on re-shaped sites

        Parameters
        ----------
        site : int
            site to be left normalized

        """
        d1 = self.Psi[site].shape[0]; d2 = self.Psi[site].shape[1]
        d3 = self.Psi[site].shape[2]; d1p = self.Psi[site+1].shape[0] 
        d2p = self.Psi[site+1].shape[1]; d3p = self.Psi[site+1].shape[2]; 
        psi_m = self.Psi[site].reshape(d1*d2, d3)
        u, d, vh = LA.svd(psi_m)
        self.Psi[site] = u[:,np.arange(0,len(d),1)].reshape(d1,d2,d3)
        psi_mp = np.diag(d) @ vh @ self.Psi[site+1].reshape(d1p, d2p*d3p) / LA.norm(d)
        self.Psi[site+1] = psi_mp.reshape(d1p, d2p, d3p)
        
    def right_canonize(self, site):
        """
        Right canonoize sites by performing SVD on re-shaped sites

        Parameters
        ----------
        site : int
            site to be right normalized

        """
        d1 = self.Psi[site].shape[0]; d2 = self.Psi[site].shape[1]
        d3 = self.Psi[site].shape[2]; d1p = self.Psi[site-1].shape[0] 
        d2p = self.Psi[site-1].shape[1]; d3p = self.Psi[site-1].shape[2]; 
        psi_m = self.Psi[site].reshape(d1, d2* d3)
        u, d, vh = LA.svd(psi_m)
        self.Psi[site] = vh[np.arange(0,len(d),1),:].reshape(d1,d2,d3)
        psi_mp = (self.Psi[site-1].reshape(d1p*d2p, d3p) @ u @ np.diag(d)) / LA.norm(d)
        self.Psi[site+1] = psi_mp.reshape(d1p, d2p, d3p)
        
    def contract_to_dense(self):
        """
        Contracts MPS into dense tensor of rank n, each dimension 2

        Returns
        -------
        contracted : array
            dense contracted state-tensor from MPS
        """
        
        t1 = np.array([1])
        contracted = np.tensordot(t1, self.Psi[0], axes=((0),(0)))
        for i in np.arange(1,self.N, 1):
            contracted = np.tensordot(contracted, self.Psi[i], axes=((i), (0)))
        contracted=np.tensordot(contracted, t1, axes=((self.N), (0)))
        return contracted
    
    def sqgate(self, site, gate):
        """
        applies single qubit gate by contracting with physical index, 

        Parameters
        ----------
        site : TYPE
            DESCRIPTION.
        gate : TYPE
            DESCRIPTION.


        """
        self.Psi[site] = np.tensordot(self.Psi[site], gate, axes=((1),(0)))
        self.Psi[site] = np.swapaxes(self.Psi[site],1,2)
        
    def twoqgate(self, control, gate):
        """
        applies two qubit gate by re-shaping MPS on control and target into 
        one site, contracting with 2 qubit gate on dim 4 physical index
        before re-shaping back into two nodes using the SVD

        Parameters
        ----------
        control : TYPE
            DESCRIPTION.
        gate : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        left_one = self.Psi[control].shape[0]
        right_two = self.Psi[control+1].shape[2]
        two_site = np.tensordot(self.Psi[control], self.Psi[control+1], axes=((2),(0)))
        two_site = np.reshape(two_site, (two_site.shape[0], 4, two_site.shape[3]))
        two_site = np.tensordot(two_site, gate, axes=((1),(0)))
        two_site = np.swapaxes(two_site, 1, 2)
        two_site = np.reshape(two_site, (two_site.shape[0]*2, 2*two_site.shape[2]))
        u, d, vh = LA.svd(two_site)
        u = u[:,np.arange(0,len(d),1)]
        self.Psi[control] = np.reshape(u, (left_one, 2, u.shape[1]))
        rmat = np.diag(d) @ vh[np.arange(0,len(d),1),:]
        self.Psi[control+1] = np.reshape(rmat, (len(d), 2, right_two) )
        
    def canonize_psi(self):
        """
        canonizes psi by SVD on each site respective to center of canonization

        """
        for i in range(self.c_center):
            self.left_canonize(i)     
        for i in range(self.N-1, self.c_center, -1):
            self.right_canonize(i)
        ### Normalize center tensor
        self.Psi[self.c_center] = self.Psi[self.c_center]/LA.norm(self.Psi[self.c_center])
        
#%%
"""
Make GHZ State
"""
N = 10
circ = Circuit(N, max_chi = 2**(N//2), init_state='zero', c_center=N-1)
circ.init_Circuit()
circ.sqgate(0, H)
for j in np.arange(0,N-1,1):
    circ.twoqgate(j, CNOT)
c_tensor = circ.contract_to_dense()
#%%
"""
Make state to test bond dim of everything... and it does. bond dimension does 
not go over the maximum. And runs pretty efficiently, compared to the qiskit
backend
"""
N = 20
circ2 = Circuit(N, max_chi = 2**(N//2), init_state='zero', c_center=N-1)
circ2.init_Circuit()
for k in range(10):
    for j in np.arange(0,N-1,1):
        circ2.sqgate(j, H)
        circ2.sqgate(j,T)
        circ2.twoqgate(j, CNOT)
        circ2.sqgate(j, H)
        circ2.sqgate(j, X)
   