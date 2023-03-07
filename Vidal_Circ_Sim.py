#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vidal_Circ_Sim.py
---------------------------------------------------------------------
Simulating quantum circuits using tensor networks, following methods
outlines in Vidal [2003]. 

Goal is to look at how information spreads in circuits of different classes, 
and how bond dimension grows as a circuit goes from being filled with the 
identity to being filled with brickwork architecture

TO-DO
    Generate random cliffords... qiskit? via tableau? lets see... 

    by Andrew Projansky - last modified February 1st
"""

import numpy as np
from numpy import linalg as LA
import random
import matplotlib.pyplot as plt
import qiskit
from quimb import ptr
from tqdm import tqdm
import time 
#import PlottingTNs
#%%
error_t = 10**(-8)
H = 1/np.sqrt(2) * np.array([[1,1],[1,-1]])
S = np.array([[1,0],[0,1j]])
T = np.array([[1,0],[0,np.exp(1j*np.pi/4)]])
X = np.array([[0,1],[1,0]])
Y = np.array([[0,-1j],[1j,0]])
Z = np.array([[1,0],[0,-1]])
CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])

def cc(mat):
    
    return np.conj(mat).T

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
def make_sim(dim, simmean, simwidth):
    """
    Makes randomly matrix in GL(N, C). This matrix can be assumed to be 
    invertible because the measure of non-invertible matrices when 
    randomly selecting from C(N) is zero

    Parameters
    ----------
    dim : int
        dimension of matrix
    simmean : int
        mean of distribution random complex variables are chosen from
    simwidth : int
        width of distribution random complex variables are chosen from

    Returns
    -------
    SM : array
        matrix in GL(N, C)

    """
    
    RR = np.random.normal(simmean, simwidth, (dim,dim))
    SM = RR + 1j * np.random.normal(simmean, simwidth, (dim,dim))
    return SM

def make_unitary(dim, simmean, simwidth):
    """
    Generates unitary matrix via QR decomposition of matrix in GL(N, C)
    See parameters above

    Returns
    -------
    U : array
        unitary array

    """
    sim = make_sim(dim, simmean, simwidth)
    Q, R = np.linalg.qr(sim)
    Etta = np.zeros((dim,dim))
    for j in range(dim):
        Etta[j,j] = R[j,j]/LA.norm(R[j,j])
    U = Q @ Etta
    return U

def dephase(unitary):
    """
    Dephases unitary, turns it from U(N) to SU(N)

    Parameters
    ----------
    unitary : array
        input matrix in U(N)

    Returns
    -------
    unitary : array
        depahsed matrix in SU(N)

    """
    
    glob = np.linalg.det(unitary)
    theta = np.arctan(np.imag(glob) / np.real(glob)) / 2
    unitary = unitary * np.exp(-1j*theta)
    if np.round(np.linalg.det(unitary)) < 0:
        unitary = unitary * 1j
    return unitary

#%%
class Experiment:
    
    def __init__(self, gates, repeats, probs, data_type, layers,
                 N, max_chi = None, in_state = 'zero', c_center=None):
        """
        Class object for initializing an experiment where over a 
        certain brickwork gate set, with bricks being non-identity only
        with a certain probability, circuits are run with callable 
        stats

        Parameters
        ----------
        gates : string
            set of gates. Clifford, Haar, or Match
        repeats : int
            number of experiments data will be averaged over
        probs : float
            probability in [0,1] for gates in brickwork to be non-identity or 
            not
        data_type : dict
            dictionary of types of data to be recorded
        layers : int
            layers in brickwork
        N : int
            Dsize of space
        max_chi : int, optional
            maximum size of bonds in TN circuit. The default is None.
        in_state : string, optional
            Initial MPS state. The default is 'zero'.
        c_center : int, optional
            site of canonization. The default is None.
            
        Attributes
        ----------
        bricks:  list
            list of brickwork pairs
        circ : Circuit
            circuit object in experiment
        data : array
            array of data values recoreded at each layer
        no_id : array
            list of non identity gates, for plotting
        id_list : array
            list of identity gates, for plotting

        """
        
        self.gates = gates
        self.repeats = repeats
        self.probs = probs
        self.data_type = data_type
        self.layers = layers
        self.N = N
        self.max_chi = max_chi
        self.in_state = in_state
        self.c_center = c_center
        self.bricks = self.make_bricks()
        self.circ = None
        self.data = np.zeros((len(data_type.keys()), layers))
        self.not_id = [[] for j in range(self.layers)]
        self.id_list = [[] for j in range(self.layers)]
        
        
    def make_bricks(self):
        """
        Makes brickwork pairs based on number of sites

        Returns
        -------
        layers : list
            list of pairs for brickwork

        """
        
        layers = [[],[]]
        for j in range(self.N//2):
            layers[0].append([2*j,2*j+1])
        for j in range((self.N-1)//2):
            layers[1].append([2*j + 1, 2*j+2])
        return layers
    
    def renyi(self, rho_red, alpha):
        """
        alpha renyi entropy, with special consideration taken for when 
        alpha is 1 (Von Neumann Entropy)

        Parameters
        ----------
        rho_red : array
            reduced density matrix for entropy to be taken over
        alpha : int
            order of entropy

        Returns
        -------
        S : float
            renyi entropy of order alpha

        """
        d, u = np.linalg.eig(rho_red); S = 0; d = d[d != 0]
        if alpha == 1:
            for eig in d:
                S += -1*eig*np.log2(eig)
        else:
            S = 1/(1-alpha) * np.log2(sum(d**(alpha)))
        return S
        
    def gate_funct(self):
        """
        Returns gate of desired gate set

        Returns
        -------
        gate : array
            two qubit gate to be applied

        """
        
        if self.gates == 'Haar':
            gate = make_unitary(4, 0, 1)
        if self.gates == 'Match':
            u1 = make_unitary(2, 0, 1); u2 = make_unitary(2, 0, 1)
            u1 = dephase(u1); u2 = dephase(u2)
            gate = np.zeros((4,4), dtype='complex')
            for i in range(2):
                for j in range(2):
                    gate[3*i, 3*j] = u1[i,j]
                    gate[i+1, j+1] = u2[i,j]
        if self.gates == 'Clifford':
            gate = qiskit.quantum_info.random_clifford(2)
        return gate
    
    def get_data(self, layer):
        """
        Generates data at each layer

        Parameters
        ----------
        layer : int
            Dcurrent layer circuit is on

        """
        ind = 0
        for key in self.data_type.keys():
            if 'bond_dim' == key:
                m = 0
                for t in self.circ.Psi:
                    if t.shape[0] > m:
                        m = t.shape[0]
                self.data[ind, layer] += m
                ind = ind + 1
            if 'vn' == key:
                final_A = self.data_type['vn']
                rho = np.outer(np.conj(self.circ.contract_to_dense()), 
                               self.circ.contract_to_dense())
                rho_red = ptr(rho, [2]*self.N, np.arange(1, final_A, 1))
                self.data[ind, layer] += self.renyi(rho_red, 1)
                ind = ind + 1
            if 'renyi' == key:
                alpha = self.data_type['renyi'][0]
                final_A = self.data_type['renyi'][1]
                rho = np.outer(np.conj(self.circ.contract_to_dense()), 
                                self.circ.contract_to_dense())
                rho_red = ptr(rho, [2]*self.N, np.arange(1, final_A, 1))
                self.data[ind, layer] += self.renyi(rho_red, alpha)
                ind = ind + 1
            
    def run_experiment(self):
        """
        Runs experiment, over a number of repeats generates circuit and 
        applies gates, taking statistics of system during process


        """
        
        for i in tqdm(range(self.repeats)):
            self.circ = Circuit(self.N, self.max_chi,
                                       self.in_state, self.c_center)
            self.circ.init_Circuit()
            for j in range(self.layers):
                for pairs in self.bricks[j%2]:
                    if self.probs > np.random.rand():
                        gate = self.gate_funct()
                        self.circ.twoqgate(pairs[0], gate)
                        self.not_id[j].append(pairs)
                    else:
                        self.id_list[j].append(pairs)
                self.get_data(j)
        self.data = self.data/self.repeats

#%%
"""
Make GHZ State
"""
N = 1000
circ = Circuit(N, init_state='zero')
circ.init_Circuit()
circ.sqgate(0, H)
st = time.time()
for j in np.arange(0,N-1,1):
    circ.twoqgate(j, CNOT)
print(time.time()-st)
#c_tensor = circ.contract_to_dense()
#%%
"""
Experiment from |000...00> state
"""
N = 10; layers = 100; repeats = 200
p_l = [0.05, 0.15, 0.3, 0.5, 0.85]
fig1, (ax1) = plt.subplots(1,1)
fig2, (ax2) = plt.subplots(1,1)
fig3, (ax3) = plt.subplots(1,1)
ax1.set_title('Haar: log2 bond_dim')
ax2.set_title('Haar: von neumann')
ax3.set_title('Haar: second renyi')
for p in p_l:
    exp1 = Experiment('Haar', repeats, p, {'bond_dim': 0, 'vn': N//2, 'renyi': [2, N//2]}, 
                      layers, N, in_state = 'zero')
    exp1.run_experiment()
    ax1.plot(np.arange(1,layers+1,1), np.log2(exp1.data[0]), label=str(p))
    ax2.plot(np.arange(1,layers//2+1,1), exp1.data[1][np.arange(0,layers,2)])
    ax3.plot(np.arange(1,layers//2+1,1), exp1.data[2][np.arange(0,layers,2)])
ax1.legend(loc="lower right")
fig1.show()
fig2.show()
fig3.show()

N = 10; layers = 100; repeats = 200
p_l = [0.05, 0.15, 0.3, 0.5, 0.85]
fig1, (ax1) = plt.subplots(1,1)
fig2, (ax2) = plt.subplots(1,1)
fig3, (ax3) = plt.subplots(1,1)
ax1.set_title('Match: log2 bond_dim')
ax2.set_title('Match: von neumann')
ax3.set_title('Match: second renyi')
for p in p_l:
    exp1 = Experiment('Match', repeats, p, {'bond_dim': 0, 'vn': N//2, 'renyi': [2, N//2]}, 
                      layers, N, in_state = 'zero')
    exp1.run_experiment()
    ax1.plot(np.arange(1,layers+1,1), np.log2(exp1.data[0]), label=str(p))
    ax2.plot(np.arange(1,layers//2+1,1), exp1.data[1][np.arange(0,layers,2)])
    ax3.plot(np.arange(1,layers//2+1,1), exp1.data[2][np.arange(0,layers,2)])
ax1.legend(loc="lower right")
fig1.show()
fig2.show()
fig3.show()

N = 10; layers = 100; repeats = 200
p_l = [0.05, 0.15, 0.3, 0.5, 0.85]
fig1, (ax1) = plt.subplots(1,1)
fig2, (ax2) = plt.subplots(1,1)
fig3, (ax3) = plt.subplots(1,1)
ax1.set_title('Clifford: log2 bond_dim')
ax2.set_title('Clifford: von neumann')
ax3.set_title('Clifford: second renyi')
for p in p_l:
    exp1 = Experiment('Clifford', repeats, p, {'bond_dim': 0, 'vn': N//2, 'renyi': [2, N//2]}, 
                      layers, N, in_state = 'zero')
    exp1.run_experiment()
    ax1.plot(np.arange(1,layers+1,1), np.log2(exp1.data[0]), label=str(p))
    ax2.plot(np.arange(1,layers//2+1,1), exp1.data[1][np.arange(0,layers,2)])
    ax3.plot(np.arange(1,layers//2+1,1), exp1.data[2][np.arange(0,layers,2)])
ax1.legend(loc="lower right")
fig1.show()
fig2.show()
fig3.show()
#%%
#%%
"""
Experiment from random single qubit product state
"""
N = 10; layers = 100; repeats = 200
p_l = [0.05, 0.15, 0.3, 0.5, 0.85]
fig1, (ax1) = plt.subplots(1,1)
fig2, (ax2) = plt.subplots(1,1)
fig3, (ax3) = plt.subplots(1,1)
ax1.set_title('Haar: log2 bond_dim')
ax2.set_title('Haar: von neumann')
ax3.set_title('Haar: second renyi')
for p in p_l:
    exp1 = Experiment('Haar', repeats, p, {'bond_dim': 0, 'vn': N//2, 'renyi': [2, N//2]}, 
                      layers, N, in_state = 'rand_loc')
    exp1.run_experiment()
    ax1.plot(np.arange(1,layers+1,1), np.log2(exp1.data[0]), label=str(p))
    ax2.plot(np.arange(1,layers//2+1,1), exp1.data[1][np.arange(0,layers,2)])
    ax3.plot(np.arange(1,layers//2+1,1), exp1.data[2][np.arange(0,layers,2)])
ax1.legend(loc="lower right")
fig1.show()
fig2.show()
fig3.show()

N = 10; layers = 100; repeats = 200
p_l = [0.05, 0.15, 0.3, 0.5, 0.85]
fig1, (ax1) = plt.subplots(1,1)
fig2, (ax2) = plt.subplots(1,1)
fig3, (ax3) = plt.subplots(1,1)
ax1.set_title('Match: log2 bond_dim')
ax2.set_title('Match: von neumann')
ax3.set_title('Match: second renyi')
for p in p_l:
    exp1 = Experiment('Match', repeats, p, {'bond_dim': 0, 'vn': N//2, 'renyi': [2, N//2]}, 
                      layers, N, in_state = 'rand_loc')
    exp1.run_experiment()
    ax1.plot(np.arange(1,layers+1,1), np.log2(exp1.data[0]), label=str(p))
    ax2.plot(np.arange(1,layers//2+1,1), exp1.data[1][np.arange(0,layers,2)])
    ax3.plot(np.arange(1,layers//2+1,1), exp1.data[2][np.arange(0,layers,2)])
ax1.legend(loc="lower right")
fig1.show()
fig2.show()
fig3.show()

N = 10; layers = 100; repeats = 200
p_l = [0.05, 0.15, 0.3, 0.5, 0.85]
fig1, (ax1) = plt.subplots(1,1)
fig2, (ax2) = plt.subplots(1,1)
fig3, (ax3) = plt.subplots(1,1)
ax1.set_title('Clifford: log2 bond_dim')
ax2.set_title('Clifford: von neumann')
ax3.set_title('Clifford: second renyi')
for p in p_l:
    exp1 = Experiment('Clifford', repeats, p, {'bond_dim': 0, 'vn': N//2, 'renyi': [2, N//2]}, 
                      layers, N, in_state = 'rand_loc')
    exp1.run_experiment()
    ax1.plot(np.arange(1,layers+1,1), np.log2(exp1.data[0]), label=str(p))
    ax2.plot(np.arange(1,layers//2+1,1), exp1.data[1][np.arange(0,layers,2)])
    ax3.plot(np.arange(1,layers//2+1,1), exp1.data[2][np.arange(0,layers,2)])
ax1.legend(loc="lower right")
fig1.show()
fig2.show()
fig3.show()
#%%
'''
Experiment - high probability, data overlayed for each family, 
rand_product start
'''
g_type = ['Haar', 'Match', 'Clifford']; p = 0.85
N = 10; layers = 100; repeats = 200
fig1, (ax1) = plt.subplots(1,1)
fig2, (ax2) = plt.subplots(1,1)
fig3, (ax3) = plt.subplots(1,1)
ax1.set_title('log2 bond_dim: p = 0.85')
ax2.set_title('von neumann: p = 0.85')
ax3.set_title('second renyi: p = 0.85')
for g_t in g_type:
    exp1 = Experiment(g_t, repeats, p, {'bond_dim': 0, 'vn': N//2, 'renyi': [2, N//2]}, 
                      layers, N, in_state = 'rand_loc')
    exp1.run_experiment()
    ax1.plot(np.arange(1,layers+1,1), np.log2(exp1.data[0]), label=g_t)
    ax2.plot(np.arange(1,layers//2+1,1), exp1.data[1][np.arange(0,layers,2)], label=g_t)
    ax3.plot(np.arange(1,layers//2+1,1), exp1.data[2][np.arange(0,layers,2)], label=g_t)
ax1.legend(loc="lower right")
ax2.legend(loc="lower right")
ax3.legend(loc="lower right")
fig1.show()
fig2.show()
fig3.show()

'''
Experiment - low probability, data overlayed for each family,
rand_product start
'''
g_type = ['Haar', 'Match', 'Clifford']; p = 0.1
N = 10; layers = 200; repeats = 200
fig1, (ax1) = plt.subplots(1,1)
fig2, (ax2) = plt.subplots(1,1)
fig3, (ax3) = plt.subplots(1,1)
ax1.set_title('log2 bond_dim: p = 0.1')
ax2.set_title('von neumann: p = 0.1')
ax3.set_title('second renyi: p = 0.1')
for g_t in g_type:
    exp1 = Experiment(g_t, repeats, p, {'bond_dim': 0, 'vn': N//2, 'renyi': [2, N//2]}, 
                      layers, N, in_state = 'rand_loc')
    exp1.run_experiment()
    ax1.plot(np.arange(1,layers+1,1), np.log2(exp1.data[0]), label=g_t)
    ax2.plot(np.arange(1,layers//2+1,1), exp1.data[1][np.arange(0,layers,2)], label=g_t)
    ax3.plot(np.arange(1,layers//2+1,1), exp1.data[2][np.arange(0,layers,2)], label=g_t)
ax1.legend(loc="lower right")
ax1.legend(loc="lower right")
ax2.legend(loc="lower right")
ax3.legend(loc="lower right")
fig1.show()
fig2.show()
fig3.show()
#%%
#%%
'''
Experiment - high probability, data overlayed for each family, 
zero state start
'''
g_type = ['Haar', 'Match', 'Clifford']; p = 0.85
N = 10; layers = 100; repeats = 200
fig1, (ax1) = plt.subplots(1,1)
fig2, (ax2) = plt.subplots(1,1)
fig3, (ax3) = plt.subplots(1,1)
ax1.set_title('log2 bond_dim: p = 0.85')
ax2.set_title('von neumann: p = 0.85')
ax3.set_title('second renyi: p = 0.85')
for g_t in g_type:
    exp1 = Experiment(g_t, repeats, p, {'bond_dim': 0, 'vn': N//2, 'renyi': [2, N//2]}, 
                      layers, N, in_state = 'zero')
    exp1.run_experiment()
    ax1.plot(np.arange(1,layers+1,1), np.log2(exp1.data[0]), label=g_t)
    ax2.plot(np.arange(1,layers//2+1,1), exp1.data[1][np.arange(0,layers,2)], label=g_t)
    ax3.plot(np.arange(1,layers//2+1,1), exp1.data[2][np.arange(0,layers,2)], label=g_t)
ax1.legend(loc="lower right")
ax2.legend(loc="lower right")
ax3.legend(loc="lower right")
fig1.show()
fig2.show()
fig3.show()

'''
Experiment - low probability, data overlayed for each family,
rand_product start
'''
g_type = ['Haar', 'Match', 'Clifford']; p = 0.1
N = 10; layers = 200; repeats = 200
fig1, (ax1) = plt.subplots(1,1)
fig2, (ax2) = plt.subplots(1,1)
fig3, (ax3) = plt.subplots(1,1)
ax1.set_title('log2 bond_dim: p = 0.1')
ax2.set_title('von neumann: p = 0.1')
ax3.set_title('second renyi: p = 0.1')
for g_t in g_type:
    exp1 = Experiment(g_t, repeats, p, {'bond_dim': 0, 'vn': N//2, 'renyi': [2, N//2]}, 
                      layers, N, in_state = 'zero')
    exp1.run_experiment()
    ax1.plot(np.arange(1,layers+1,1), np.log2(exp1.data[0]), label=g_t)
    ax2.plot(np.arange(1,layers//2+1,1), exp1.data[1][np.arange(0,layers,2)], label=g_t)
    ax3.plot(np.arange(1,layers//2+1,1), exp1.data[2][np.arange(0,layers,2)], label=g_t)
ax1.legend(loc="lower right")
ax1.legend(loc="lower right")
ax2.legend(loc="lower right")
ax3.legend(loc="lower right")
fig1.show()
fig2.show()
fig3.show()
#%%
u1 = make_unitary(2, 0, 1); u2 = make_unitary(2, 0, 1)
u1 = dephase(u1); u2 = dephase(u2)
gate = np.zeros((4,4), dtype='complex')
for i in range(2):
    for j in range(2):
        gate[3*i, 3*j] = u1[i,j]
        gate[i+1, j+1] = u2[i,j]