   # -*- coding: utf-8 -*-
"""
DMRG.py
---------------------------------------------------------------------
File for understanding DMRG through example of calculating ground state
energy of XX chain with 1 site DMRG

Q: For example encoded below, should converge to about -63; I'm off by exactly
a factor of two, and am unsure where

    by Andrew Projansky - last modified 1/30/2023
    
    Initialization of Hamiltonian and of MPS from 2 site DMRG example 
    on tensors.net ; by Glen Evenbly
"""

#### Preamble
import numpy as np
from numpy import linalg as LA

##### Set bond dimensions and number of sites
chi = 16;
Nsites = 50;

numsweeps = 5 # number of DMRG sweeps
lanit = 1 # iterations of Lanczos method
krydim = 4 # dimension of Krylov subspace

"""

Initialization of XX Hamiltonian

For single site DMRG, Hamiltonians initialized via encoding into a bulk
operator (Schollwoeck [2011] 6.1). Hamiltonian encoded below is in creation
annihilation basis - as ground state energy is basis invariant, we can choose
basis such that bulk operator is most simply represented. 

To encode arbitrary hamiltonian, take bulk operator of dimxdim and 
re-shape into (dim/2, 2, dim/2, 2), before swapping center axes, so indice 
ordering of tensor has down physical index as axes 2 (starting from 0) and 
up physical index is 3

tensors ML and MR are those which represent the boundary of the Hamiltonian;
to make the bulk encoding correct, edge tensors are initialize to contract
operators properly; can also represent environmental system outside sites
of focus

"""
d = 2
sP = np.sqrt(2)*np.array([[0, 0],[1, 0]])
sM = np.sqrt(2)*np.array([[0, 1],[0, 0]])
sX = np.array([[0, 1], [1, 0]])
sY = np.array([[0, -1j], [1j, 0]])
sZ = np.array([[1, 0], [0,-1]])
sI = np.array([[1, 0], [0, 1]])
M = np.zeros([4,4,d,d]);
M[0,0,:,:] = sI; M[3,3,:,:] = sI
M[0,1,:,:] = sM; M[1,3,:,:] = sP
M[0,2,:,:] = sP; M[2,3,:,:] = sM
ML = np.array([1,0,0,0]).reshape(4,1,1) #left MPO boundary - environment 
MR = np.array([0,0,0,1]).reshape(4,1,1) #right MPO boundary - environment 


def Lanzcos(H, rjp, krydim, its):
    """
    Uses Lanzcos algorithm to build tridiagonal representation of H 
    whose lowest eigenvalue approximates the lowest eigenvalue of H on the 
    vector rjp. 
    
    Diagonalization of Lanzcos H is exact... should exist quicker way 
    for tridiagonal. Possibly power method of inverse? Something to look into

    Parameters
    ----------
    H : array
        Hamiltonian operator built from contracting tensor network over all 
        sites besides site being optimized, reshaped into a matrix
    rjp : array
        tensor at site to be optimized, reshaped into a vector 
    krydim : int
        dimension of krylov space/of Lanzcos hamiltonian
    its : int
        number of iterations of lanzcos method before returning vector

    Returns
    -------
    rjp : array
        vector corresponding to lowest eigenstate of lanzcos hamiltonian

    """
    k_l = [] ; a_l = [] ;b_l = [] ; qc = rjp/LA.norm(rjp) ; qp = 0 ; beta = 0
    for k in range(1, krydim+1):
        k_l.append(qc)
        ze = H @ qc
        alp = float(qc.T @ ze)
        z = ze - alp * qc - beta*qp
        beta = LA.norm(z)
        a_l.append(alp) ; b_l.append(beta)
        if beta == 0:
            break
        qp = qc ; qc = z/beta
    
    dim = len(a_l)
    b_l = b_l[:dim-1]
    H_eff = np.diag(a_l) + np.diag(b_l, k=-1) +  np.diag(b_l, k=1)
    d, u = LA.eig(H_eff)
    effsort = np.argsort(d)
    eff_e = u.T[effsort[0]]
    V = np.array(k_l).T
    rjp = (V @ eff_e).T
            
    return rjp

def contract(k, L, R):
    """
    For each optimization step, given site k, contracts transfer matrices
    in L and R with bulk hamiltonian to get 6 legged effective Hamiltonian 
    at site k

    Parameters
    ----------
    k : site number
        site being contracted around
    L : array
        array of transfer operators from left normalzied sites
    R : array
        array of transfer operators from right normalized sites

    Returns
    -------
    H : array
        6 legged tensor which acts on site k

    """
    if k != 0:
        LT = np.tensordot(L[0],L[1], axes = ((1,0,2),(0,2,4)))
        for x in range(k-1):
            LT = np.tensordot(LT, L[x+2], axes=((0,1,2),(0,2,4)))
    else:
        LT = np.swapaxes(L[0],0,1)
    
    if k != Nsites-1:
        RT = np.tensordot(R[Nsites-1], R[Nsites], axes=((1,3,5),(1,0,2)))
        for y in range(Nsites-2, k, -1):
            RT = np.tensordot(R[y], RT, axes=((1,3,5),(0,1,2)))
    else:
        RT = np.swapaxes(R[Nsites], 0,1)
    h1 = np.tensordot(LT, M, axes=((1),(0)))
    H = np.tensordot(h1, RT, axes=((2),(1)))
    return H


def init__ex_lanzcos(H, Ak):
    """
    Initializes and executes lanzcos method, by reshaping inputs into 
    square matrices and vectors and then calling the Lanzcos function

    Parameters
    ----------
    H : array
        6 leg tensor of the effective hamiltonian on site k
    Ak : array
        tensor of MPS at site k

    Returns
    -------
    upPsi : array
        approximated lowest eigenvector of H
    d1 : int
        dimension of left bond of tensor Ak
    d2 : int
        dimension of physical bond of tensor Ak
    d3 : int
        dimension of right bond of tensor Ak

    """

    re_H = np.transpose(H, (1,3,5,0,2,4))
    d1 = re_H.shape[0] ; d2 = re_H.shape[1]; d3 = re_H.shape[2]; hd = d1*d2*d3
    Hmat = np.reshape(re_H, (hd,hd))
    Psi = np.reshape(Ak, (hd,1))
    upPsi = Lanzcos(Hmat, Psi, krydim, lanit)
    return upPsi, d1, d2, d3
#%%  
#### Initialize random MPS tensor with maximum bond chi
A = [0 for x in range(Nsites)]
A[0] = np.random.rand(1,d,d)
for k in range(1,Nsites):
    A[k] = np.random.rand(A[k-1].shape[2],d,min(min(chi,A[k-1].shape[2]*d),d**(Nsites-k-1)))

"""

Initializes tensors for holding transfer operators, split into left and right 
normalized operators. AReshapes MPS tensors into canonical form for DMRG, and 
builds rank 6 (edge tensors rank 3) transfer operators

Some strange things going on with canonization... when chi is large enough 
normalization of tensors isn't quite correct it seems... is this cause of my 
error?

"""
L = [0 for x in range(Nsites+1)]
L[0] = ML
R = [0 for x in range(Nsites+1)]
R[Nsites] = MR     

for j in range(Nsites-1, 0, -1):
    b_left = A[j].shape[0] ; b_right = A[j].shape[2]
    u, sch, vh = LA.svd(A[j].reshape(b_left, d*b_right), full_matrices = False)
    A[j] = vh.reshape(b_left, d, b_right)
    b_lefts = A[j-1].shape[0] ; b_rights = A[j-1].shape[2]
    A[j-1] = (A[j-1].reshape(b_lefts*d, b_rights) @ u @ np.diag(sch)) / LA.norm(sch)
    A[j-1] = A[j-1].reshape(b_lefts, d, b_rights)
    t1 = np.tensordot(A[j], M, axes=((1),(2)))
    R[j] = np.tensordot(t1, np.conj(A[j]), axes=((4),(1)))
    
#%%
    
"""

Conducts sweeps for optimizing tensors at each site, starting with site 0 and
sweeping towards the right

"""
for j in range(numsweeps):
    #Left Sweep
    for k in range(Nsites-1):
        H = contract(k,L,R)
        upPsi, d1, d2, d3 = init__ex_lanzcos(H, A[k])
        """
        Re-shapes updated Psi back into site tensor, before svd to make new 
        site left normalized. When sent through svd, reshaped with physical 
        index going left, so that when reshaped back into rank 3 tensor has 
        correct ordered information, and right normalized part of svd can be 
        reshaped into node to the right 
        """
        A[k] = np.reshape(upPsi, (d1, d2, d3))
        b_left = A[k].shape[0] ; b_right = A[k].shape[2]
        u, sch, vh = LA.svd(A[k].reshape(b_left*d, b_right), full_matrices = False)
        A[k] = u.reshape(b_left, d, b_right)
        b_lefts = A[k+1].shape[0] ; b_rights = A[k+1].shape[2]
        A[k+1] = np.diag(sch) @ vh @ (A[k+1].reshape(b_lefts, d*b_rights)) / LA.norm(sch)
        A[k+1] = A[k+1].reshape(b_lefts, d, b_rights)
        
        """
        Updates arrays L and R, for optimization of next node
        is this the cause of error? 
        """
        t1 = np.tensordot(A[k], M, axes=((1),(2)))
        L[k+1] = np.tensordot(t1, np.conj(A[k]), axes=((4),(1)))
        R[k+1] = 0
    #Right Sweep
    for k in range(Nsites-1, 0, -1):
        H = contract(k,L,R)
        upPsi, d1, d2, d3 = init__ex_lanzcos(H, A[k])
        A[k] = np.reshape(upPsi, (d1, d2, d3))
        b_left = A[k].shape[0] ; b_right = A[k].shape[2]
        u, sch, vh = LA.svd(A[k].reshape(b_left,d*b_right), full_matrices = False)
        A[k] = vh.reshape(b_left, d, b_right)
        b_lefts = A[k-1].shape[0] ; b_rights = A[k-1].shape[2]
        A[k-1] = (A[k-1].reshape(b_lefts*d, b_rights) @ u @ np.diag(sch)) / LA.norm(sch)
        A[k-1] = A[k-1].reshape(b_lefts, d, b_rights)
        t1 = np.tensordot(A[k], M, axes=((1),(2)))
        R[k] = np.tensordot(t1, np.conj(A[k]), axes=((4),(1)))
        L[k+1] = 0
        
    #### Uncomment block below to get energy after each sweep    
    #"""    
    LS = np.tensordot(L[0],L[1], axes=((1,0,2),(0,2,4)))
    RS = np.tensordot(R[Nsites-1], R[Nsites], axes=((1,3,5),(1,0,2)))
    for y in range(Nsites-2, 0, -1):
        RS = np.tensordot(R[y], RS, axes=((1,3,5),(0,1,2)))
    E = float(np.tensordot(LS, RS, axes=((0,1,2),(0,1,2))))
    print(E)
    #"""
#%%
        
"""
Final contraction of the updated tensor network, contracting all left 
normalized transfer operators (left boundary, first tensor) and right normalized
tensors (sites 2-N, and right boundary). 
"""

LS = np.tensordot(L[0],L[1], axes=((1,0,2),(0,2,4)))
RS = np.tensordot(R[Nsites-1], R[Nsites], axes=((1,3,5),(1,0,2)))
for y in range(Nsites-2, 0, -1):
    RS = np.tensordot(R[y], RS, axes=((1,3,5),(0,1,2)))
E = float(np.tensordot(LS, RS, axes=((0,1,2),(0,1,2))))
