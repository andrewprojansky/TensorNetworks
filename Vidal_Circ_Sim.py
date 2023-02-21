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
from scipy.stats import unitary_group

error_t = 10**(-8)
H = 1/np.sqrt(2) * np.array([[1,1],[1,-1]])
S = np.array([[1,0],[0,1j]])
T = np.array([[1,0],[0,np.exp(1j*np.pi/4)]])
X = np.array([[0,1],[1,0]])
Y = np.array([[0,-1j],[1j,0]])
Z = np.array([[1,0],[0,-1]])
CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
#%%
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
    
    def __init__(self, N=10, max_chi = None,
                init_state = 'zero', c_center=None):
        self.N = N
        if max_chi == None: max_chi = 2**(N//2)
        else: self.max_chi = max_chi
        self.init_state = init_state
        self.Psi = [0 for x in range(N)]
        if c_center==None: self.c_center = N-1
        else: self.c_center = c_center
        
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
        d = self.trun_d(d)
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
        d = self.trun_d(d)
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
        d = self.trun_d(d)
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
        
    def trun_d(self, d):
        """
        Truncates singular values based on error threshold 
        ... need to implement max_chi truncation as well

        Parameters
        ----------
        d : array
            array of non truncated singular values

        Returns
        -------
        d : array
            Dtruncated vector of singular values

        """
        
        for i in range(len(d)-1,-1,-1):
            if d[i] > error_t:
                d = d[:i+1]
                break
        return d
        
#%%
"""
Make GHZ State
"""
N = 10
circ = Circuit(N, init_state='zero')
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
circ2 = Circuit(N, init_state='zero')
circ2.init_Circuit()
for k in range(10):
    for j in np.arange(0,N-1,1):
        circ2.sqgate(j, H)
        circ2.sqgate(j,T)
        circ2.twoqgate(j, CNOT)
        circ2.sqgate(j, H)
        circ2.sqgate(j, X)
#%%
"""
Testing Random Clifford Circuits; do statistics match statistics from qiskit
backend? 
"""
repeats=20

def run_circ(p_cnot,N):
    h_c = 0; s_c = 0; cnot_c = 0; p_single =  1-p_cnot
    # Construct quantum circuit
    circclif = Circuit(N, init_state='zero')
    circclif.init_Circuit()
    
    #layers = N**2
    layers= N**2
    for j in range(layers):
        i_banned = []
        for i in range(N):
            two_q = random.random()
            if two_q < p_cnot and i < N-1 and i not in i_banned:
                circclif.twoqgate(i, CNOT); i_banned.append(i+1) ; cnot_c += 1
            elif two_q < p_cnot + p_single/2  and i not in i_banned:
                circclif.sqgate(i,H); h_c+= 1
            elif two_q > p_cnot + p_single/2 and i not in i_banned:
                circclif.sqgate(i, S); s_c += 1
                
    m_l = 0
    for t in circclif.Psi:
        if t.shape[0]  > m_l:
            m_l = t.shape[0]
    
    return cnot_c, m_l

avg_bond = []
avg_cnot_c = []
perc_l = []
ent_l = []
for j in tqdm(np.arange(0,100,1)):
    p_cnot = 0.01*j
    avg_cnot = 0 ; avg_max_b = 0
    for i in range(repeats):
        cnot_c, m_l = run_circ(p_cnot,10)
        avg_cnot += cnot_c ; avg_max_b += m_l
    avg_cnot = np.round(avg_cnot/repeats) ; avg_max_b = avg_max_b/repeats
    avg_bond.append(avg_max_b) ; avg_cnot_c.append(avg_cnot)
    perc_l.append(p_cnot)
    
plt.plot(perc_l, avg_bond)
plt.xlabel('Percentage for random CHP gate to be CNOT')
plt.ylabel('Avg bond dimension')
plt.suptitle('% CNOT v Avg bond dim (100 trials per %)', fontsize=14)
plt.title('Random Initial Two Qubit Product State', fontsize=10)
plt.show()
#%%
"""
Do the above with Match Gates
"""
def make_sim(dim, simmean, simwidth):
    
    RR = np.random.normal(simmean, simwidth, (dim,dim))
    SM = RR + 1j * np.random.normal(simmean, simwidth, (dim,dim))
    return SM

def make_unitary(dim, sim):
    Q, R = np.linalg.qr(sim)
    Etta = np.zeros((dim,dim))
    for j in range(dim):
        Etta[j,j] = R[j,j]/LA.norm(R[j,j])
    U = Q @ Etta
    return U

def rand_match():
    sim1 = make_sim(2, 0, 1); u1 = make_unitary(2, sim1)
    sim2 = make_sim(2, 0, 1); u2 = make_unitary(2, sim2)
    match = np.zeros((4,4) , dtype='complex')
    for j in range(2):
        for k in range(2):
            match[3*j,3*k] = u1[j,k]
            match[j+1, k+1] = u2[j,k]
            
    return match
#%%
repeats=20

def run_circ(p_match,N):
    # Construct quantum circuit
    n_match=0
    circmatch = Circuit(N, init_state='zero')
    circmatch.init_Circuit()
    
    #layers = N**2
    layers= N**2
    for j in range(layers):
        i_banned = []
        for i in range(N):
            two_q = random.random()
            if two_q < p_match and i < N-1 and i not in i_banned:
                circmatch.twoqgate(i, rand_match()); i_banned.append(i+1)
                n_match = n_match+1
            elif two_q > p_match and i not in i_banned:
                sim1 = make_sim(2, 0, 1); u1 = make_unitary(2, sim1)
                circmatch.sqgate(i, u1)
                
                
    m_l = 0
    for t in circmatch.Psi:
        if t.shape[0]  > m_l:
            m_l = t.shape[0]
    
    return n_match, m_l

avg_bond = []
avg_match = []
perc_l = []
ent_l = []
for j in tqdm(np.arange(0,80,1)):
    p_match = 0.01*j
    avg_m = 0 ; avg_max_b = 0
    for i in range(repeats):
        m_c, m_l = run_circ(p_match,10)
        avg_m += m_c ; avg_max_b += m_l
    avg_cnot = np.round(avg_m/repeats) ; avg_max_b = avg_max_b/repeats
    avg_bond.append(avg_max_b) ; avg_match.append(avg_m)
    perc_l.append(p_match)
    
plt.plot(perc_l, avg_bond)
plt.xlabel('Percentage for random gate to be match gate')
plt.ylabel('Avg bond dimension')
plt.suptitle('% Match v Avg bond dim (100 trials per %)', fontsize=14)
plt.title('From Zero State', fontsize=10)
plt.show()