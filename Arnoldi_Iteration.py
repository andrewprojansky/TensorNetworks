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
    """
    Shorthand for complex conjugate transpose

    Parameters
    ----------
    v : array
        input vector or matrix for complex conjugate transpose to be taken

    """
    
    return np.conj(v.T)

def tol_check(mat, x, eigs, tol, k):
    """
    Checks tolerance by checking if max error from ritz eigenvalue being 
    an actual eigenvalue is greater than some error tolerance

    Parameters
    ----------
    mat : array
        matrix arnoldi scheme is applied to
    x : array
        list of ritz eigenvectors
    eigs : array
        list of ritz eigenvalues
    tol : float
        tolerance for error
    k : int
        size of initial orthonormal Krylov space

    Returns
    -------
    bool
        boolean for if error is below tolerance or not

    """
    
    t = [(LA.norm(mat @ x[i] - eigs[i] * x[i])) for i in range(k)]
    if max(t) > tol: return True
    else: return False
    
def sort_eigs(sort_crit, new_eigs, k, p):
    """
    sorts eigenvalues depending on sorting criteria, and returns 
    p undesired eigenvalues for shifts to occur on
    
    sort_crit can take values
        'Max Norm', 'Min Norm', 'Max Real', 'Min Real', 'Max Imag', 'Min Imag'

    Parameters
    ----------
    sort_crit : string
        sorting criterion, dictates what eigenvalues are shifts 
    new_eigs : array
        array of input eigenvalues
    k : int
        size of extended orthonormal Krylov space
    p : int
        size of extension

    Returns
    -------
    shifts: array
        eigenvalues for shifts to be applied

    """
    
    if sort_crit == 'Max Norm' or sort_crit == 'Min Norm':
        norms = [LA.norm(new_eigs[x]) for x in range(k)]
        il = list(np.argsort(norms))
        if sort_crit == 'Max Norm': il = il[:p]
        if sort_crit == 'Min Norm': il = il[k-p:] 
    if sort_crit == 'Max Real' or sort_crit == 'Min Real':
        reals = [np.real(new_eigs[x]) for x in range(k)]
        il = list(np.argsort(reals))
        if sort_crit == 'Max Real': il = il[:p]
        if sort_crit == 'Min Real': il = il[k-p:] 
    if sort_crit == 'Max Imag' or sort_crit == 'Min Imag':
        imags = [np.imag(new_eigs[x]) for x in range(k)]
        il = list(np.argsort(imags))
        if sort_crit == 'Max Imag': il = il[:p]
        if sort_crit == 'Min Imag': il = il[k-p:]
    shifts = new_eigs[il]
    return shifts
    
def Arnoldi_Base(mat, qinit, k, q, H):
    """
    Basic Arnoldi Algorithm, with the ability to 
    extended an initial Arnoldi pass to get more ritz
    eigenvalue/vector pairs

    Parameters
    ----------
    mat : array
        matrix eigenvalues are being estimated for
    qinit : array
        initial q vector for Arnoldi or extending Arnoldi
    k : int
        dimension of Krylov space 
    q : list
        list of arrays of orthonormal vectors q; defines matrix Q
    H : array
        matrix whose eigenvalues approximate those of mat

    Returns
    -------
    q : list
        list of arrays of orthonormal vectors q; defines matrix Q
    H : array
        matrix whose eigenvalues approximate those of mat

    """
    
    ql = len(q)
    q.append(qinit)
    for j in range(ql,k):
        r = mat @ q[j]
        for i in range(j+1):
            H[i,j] = ct(q[i]) @ r ; r = r - H[i,j]*q[i]
        H[j+1,j] = LA.norm(r)
        if H[j+1, j] == 0:
            return q, H[np.arange(0,j,1):]
        q.append(r/H[j+1, j])
    return q, H

def IRA(mat, qinit, k, q, H, p, tao, sort_crit):
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
    check = tol_check(mat, x, eigs, tao, k)
    #adjust check to do 5 iterations
    check = 0
    while check != 10:
        #shift this block down when not testing stuff... 
        H = np.pad(H, [(0,p+1),(0,p+1)], mode='constant', constant_values=0)
        k=k+p; q, H = Arnoldi_Base(mat, qinit, k, q, H)
        qinit = q[k] ; q = q[:k]; H = H[:k, :k]
        new_eigs, unew = np.linalg.eig(H)
        shifts = sort_eigs(sort_crit, new_eigs, k, p)
        e_kpp = np.zeros(k); e_kpp[k-1] = 1; 
        #gets shifts right, but seemingly doesn't shift them? or shifts
        #don't work? here is where error is
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
        x =  evecs.T @ q; q = list(q)
        #check = tol_check(mat, x, eigs, tao, k)
        check = check + 1
    return eigs, x
        
        
#%%
#Base Arnoldi with seperated largest eigenvalue
dim = 20; k = 5
mat = np.random.rand(dim,dim) + 1j*np.random.rand(dim,dim)
x = np.random.rand(dim)
q = [] ; H = np.zeros((k+1,k), dtype='complex')
ex_eigs, u = np.linalg.eig(mat)
q, H = Arnoldi_Base(mat, x, k, q, H)
Qk = np.array(q)[:k].T ; H = H[:k, :k]
ap_eigs, uprime = np.linalg.eig(H)

plt.plot(np.real(ex_eigs), np.imag(ex_eigs), 'x')
plt.plot(np.real(ap_eigs), np.imag(ap_eigs), 'o', alpha=0.6)
plt.show()
#%%
#Base Arnoldi with clustered largest eigenvalue
dim = 20; k = 5
mat = np.random.rand(dim,dim)+ 1j * (np.random.rand(dim,dim))
mat = mat - (np.random.rand(dim,dim)+ 1j * (np.random.rand(dim,dim)))
x = np.random.rand(dim)
ex_eigs, u = np.linalg.eig(mat)
q = [] ; H = np.zeros((k+1,k), dtype='complex')
q, H = Arnoldi_Base(mat, x, k,q,H)
Qk = np.array(q)[:k].T ; H = H[:k, :k]
ap_eigs, uprime = np.linalg.eig(H)

plt.plot(np.real(ex_eigs), np.imag(ex_eigs), 'x')
plt.plot(np.real(ap_eigs), np.imag(ap_eigs), 'o', alpha=0.6)
plt.show()
#%%
#Extended Arnoldi- some k, extend to larger k
d = 12; 
k_range = np.arange(2,5,1)
x = np.random.rand(d); qinit = x/LA.norm(x)
mat = np.random.rand(d,d)+ 1j * (np.random.rand(d,d))
mat = mat - (np.random.rand(d,d)+ 1j * (np.random.rand(d,d)))
mat = mat + ct(mat)
for k in k_range:
    q = [] ; H = np.zeros((k+1,k), dtype='complex')
    qr, H = Arnoldi_Base(mat, qinit, k, q, H)
    qinitnew = qr[k] ; Q = qr[:k]
    ap_eigs, uprime = np.linalg.eig(H[:k,:k])
    plt.plot([k-1] * k, ap_eigs, 'x')
ex_eigs, u = np.linalg.eig(mat)
plt.plot([k]*d, ex_eigs, 'o')
plt.show()

qinit = x/LA.norm(x)
k = 2
q = [] ; H = np.zeros((3,2), dtype='complex')
q, H = Arnoldi_Base(mat, qinit, k, q, H)
qinit = q[k] ; q = q[:k]
ap_eigs, apu = np.linalg.eig(H[:k,:k])
plt.plot([k-1] * k, ap_eigs, 'x')
H = np.pad(H, [(0,1),(0,1)], mode='constant', constant_values=0)
for k in np.arange(3,5,1):
    q, H = Arnoldi_Base(mat, qinit, k, q, H)
    qinit = q[k] ; q = q[:k]
    ap_eigs, uprime = np.linalg.eig(H[:k,:k])
    H = np.pad(H, [(0,1),(0,1)], mode='constant', constant_values=0)
    plt.plot([k-1] * k, ap_eigs, 'x')
plt.plot([k]*d, ex_eigs, 'o')
plt.show()

#%%
#Implicitly Restarted Arnoldi - THIS IS AWESOME
tao = 0.01
d = 10; k = 3; p = 3
mat = np.random.rand(d,d)+ 1j * (np.random.rand(d,d))
mat = mat - (np.random.rand(d,d)+ 1j * (np.random.rand(d,d)))
mat = mat + ct(mat)
ex_eigs, u = np.linalg.eig(mat)
x = np.random.rand(d)
qinit = x/LA.norm(x)
q = [] ; H = np.zeros((k+1,k), dtype='complex')
eigs, xvs = IRA(mat, qinit, k, q, H, p, tao, 'Min Real')

qinit = x/LA.norm(x)
q = [] ; H = np.zeros((k+1,k), dtype='complex')
q, H = Arnoldi_Base(mat, qinit, k, q, H)
ei, um = np.linalg.eig(H[:k,:k])

plt.plot([0]*d, np.real(ex_eigs), 'x')
plt.plot([1]*k, np.real(ei), 'o')
plt.plot([2]*k, np.real(eigs), 'o')

plt.show()