#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmarking iterative eigensolvers on hermitian 
and non hermitian matrices with real spectra

@author: andrewprojansky
"""

from Arnoldi_Iteration import *
import numpy as np
import scipy.linalg as LA
import matplotlib.pyplot as plt
import time
#%%
def IRAu(mat, qinit, k, q, H, p, sort_crit, m_eig):
    """
    Implicitly Restarted Arnoldi: From a base arnoldi run, implicitly
    restarts process by applying polynomial in undesired eigenvalues as 
    shifts to initial vector,  projecting v0 only towards desired 
    eigenvectors until eigenvalues/vectors converge to some tolerance 

    Parameters
    ----------
    mat : array
        matrix which we use arnoldi scheme to calculate eigenvalues
    qinit : array
        inital krylov vector, or initial vector the space will be extended on
    k : int
        dimension of Krylov space
    q : list
        list of orthonormal krylov vectors
    H : array
        matrix whose eigenvalues/vectors approximate those of mat
    p : int
        size of extension of krylov space
    tao : float
        tolerance threshold
    sort_crit : string
        criterion for sorting eigenvalues and determining shifts to apply

    Returns
    -------
    eigs : array
        set of eigenvalues
    x : array
        set of eigenvectors

    """
    q, H = Arnoldi_Base(mat, qinit, k, q, H)
    qinit = q[k] ; q = q[:k]; 
    eigs, evecs = np.linalg.eig(H[:k,:k])
    x = evecs.T @ np.array(q); 
    check = 1
    me = min(eigs)
    
    meV = []
    meV.append(me)
    
    while LA.norm(me - m_eig) > 10**(-3) and check < 100:
        #shift this block down when not testing stuff... 
        H = np.pad(H, [(0,p+1),(0,p+1)], mode='constant', constant_values=0)
        k=k+p; q, H = Arnoldi_Base(mat, qinit, k, q, H)
        qinit = q[k] ; q = q[:k]; H = H[:k, :k]
        new_eigs, unew = np.linalg.eig(H)
        shifts = sort_eigs(sort_crit, new_eigs, k, p)
        e_kpp = np.zeros(k); e_kpp[k-1] = 1; 
        for s in shifts:
            Qr, qR = np.linalg.qr(H - s * np.identity(k))
            H = ct(Qr) @ H @ Qr
            q = (np.array(q).T @ Qr).T
            e_kpp = ct(e_kpp) @ Qr
        k = k-p
        q = np.array(q)
        qinit = q.T[:,k] * H[k+1, k] + qinit * e_kpp[k]
        q = q[:k]; H = H[:k,:k]
        eigs, evecs = np.linalg.eig(H)
        me = min(eigs)
        
        meV.append(me)
        
        x =  evecs.T @ q; q = list(q)
        check = check + 1
    return eigs, x, check, meV
    
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

def AB(mat, iv, k):
    qH = [] ; HH = np.zeros((k+1,k), dtype='complex')
    qH, HH = Arnoldi_Base(mat, iv, k, qH, HH)
    ap_eigs, uprime = np.linalg.eig(HH[:k,:k])
    est_min = min(ap_eigs)
    init_v = (uprime.T @ qH[:k])[np.argsort(ap_eigs)[0]]
    return est_min, init_v
    
def test_arnoldi_base(H_mat, NH_mat, d, init_v, k, m_eig):
    
    #H_mat
    est_min = 100000; Hcount=0; H_start = time.time(); iv = init_v
    while LA.norm(est_min - m_eig) > 10**(-3) and Hcount < 100:
        est_min, iv = AB(H_mat, iv, k)
        Hcount += 1
    H_time = H_start - time.time()

    #NH_mat
    est_min = 100000; NHcount=0; NH_start = time.time(); iv = init_v
    
    est_V = []
    
    while LA.norm(est_min - m_eig) > 10**(-3) and NHcount < 100:
        est_min, iv = AB(NH_mat, iv, k)
        NHcount += 1
        
        est_V.append(est_min)
        
    NH_time = NH_start - time.time()
    
    return H_time, NH_time, Hcount, NHcount, est_V

def test_IRA(H_mat, NH_mat, d, init_v, k, p, meig):
    
    #H_mat
    H_start = time.time(); iv = init_v
    qH = [] ; HH = np.zeros((k+1,k), dtype='complex')
    eigs, x, Hcount, meV = IRAu(H_mat, iv, k, qH, HH, p, 'Min Real', meig)
    H_time = H_start - time.time()
    
    NH_start = time.time(); iv = init_v
    qNH = [] ; HNH = np.zeros((k+1,k), dtype='complex')
    eigs, x, NHcount, meV = IRAu(NH_mat, iv, k, qNH, HNH, p, 'Min Real', meig)
    NH_time = NH_start - time.time()
    
    return H_time, NH_time, Hcount, NHcount, meV
    
"""
Experiment 1: Same Eig distribution, diff dimensions
"""
dim = np.arange(10,50,5)
eigmean = 0; eigwidth=0.5
simmean=0; simwidth = 1
times = np.zeros((4,len(dim)))
reps = np.zeros((4, len(dim)))
dist = np.zeros(len(dim))
k = 5
p = 3

for d in dim:
    ind = d//5-2
    eig = np.random.normal(eigmean, eigwidth, d)
    meig = min(eig)
    diff = eig[np.argsort(eig)[1]] - meig
    dist[ind] = diff
    sim = make_sim(d, simmean, simwidth)
    invsim = np.linalg.inv(sim)
    U = make_unitary(d, sim)
    Herm = U @ np.diag(eig) @ ct(U)
    Psuedo = sim @ np.diag(eig) @ invsim
    init_v = np.random.rand(d)
    
    th, tnh, repsh, repsnh, est_V = test_arnoldi_base(Herm, Psuedo,
                                               d, init_v, k, meig)
    th2, tnh2, repsh2, repsnh2, meV = test_IRA(Herm, Psuedo, d, init_v, k, p, meig)

    times[0,ind] = th;  times[1,ind] = tnh
    times[2,ind] = th2;  times[3,ind] = tnh2
    reps[0,ind] = repsh;  reps[1,ind] = repsnh
    reps[2,ind] = repsh2;  reps[3,ind] = repsnh2
    
    '''
    plt.plot(np.arange(0,repsnh, 1), est_V, 'x')
    plt.plot(np.arange(0,repsnh2, 1), meV, 'o', alpha=0.5)
    plt.show()
    '''

plt.plot(dist, reps[0], 's', color='blue')
plt.plot(dist, reps[1], 's', color='red')
#plt.plot(dist, reps[2], 's', color='black')
#plt.plot(dist, reps[3], 's', color='green')
plt.show()

plt.plot(dim, reps[0], 's', color='blue')
plt.plot(dim, reps[1], 's', color='red')
#plt.plot(dim, reps[2], 's', color='black')
#plt.plot(dim, reps[3], 's', color='green')
plt.show()
#%%
"""
Experiment 2: Same dimension, diff eig distributions
"""
dim = 50
diff = np.arange(0.5,0,-0.01)
simmean=0; simwidth = 1
times = np.zeros((4, len(diff)))
reps = np.zeros((4, len(diff)))
k = 5
p = 3

for d in range(len(diff)):
    eig = np.arange(-dim//2, dim//2, 1) * diff[d]
    meig = min(eig)
    sim = make_sim(dim, simmean, simwidth)
    invsim = np.linalg.inv(sim)
    U = make_unitary(dim, sim)
    Herm = U @ np.diag(eig) @ ct(U)
    Psuedo = sim @ np.diag(eig) @ invsim
    init_v = np.random.rand(dim)
    
    th, tnh, repsh, repsnh, est_V = test_arnoldi_base(Herm, Psuedo,
                                               dim, init_v, k, meig)
    th2, tnh2, repsh2, repsnh2, meV = test_IRA(Herm, Psuedo, dim, init_v, k, p, meig)

    times[0,d] = th;  times[1,d] = tnh
    times[2,d] = th2;  times[3,d] = tnh2
    reps[0,d] = repsh;  reps[1,d] = repsnh
    reps[2,d] = repsh2;  reps[3,d] = repsnh2
    
plt.plot(diff, reps[0], 's', color='blue')
plt.plot(diff, reps[1], 's', color='red')
plt.show()

plt.plot(diff, reps[2], 's', color='black')
plt.plot(diff, reps[3], 's', color='green')
plt.show()