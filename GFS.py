# -*- coding: utf-8 -*-
"""
GFS.py

Code for understanding Gaussian Fermionic States, and numerical methods 
for computing statsitics about quadradic fermionic Hamiltonians

By Andrew Projansky
12/21/2022
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.linalg import dft

Id = np.identity(2)
X = np.array([[0,1],[1,0]])

#%%
def an(num):
    """
    shorthand code for taking the square of the norm of complex number

    Parameters
    ----------
    num : float, complex
        number for norm squared to be taken

    Returns
    -------
    float, real
        complex conjugate

    """
    
    return np.absolute(num)**2

def cc(mat):
    """

    Parameters
    ----------
    mat : array
        matrix to be complex conjugated

    Returns
    -------
    array
        complex conjugate of array

    """
    return np.conjugate(np.transpose(mat))

        
def init_mats(dim, rand_H, A, B, NN = False):
    """
    Creates a complex hermetian matrix and a complex skew symmetric matrix
    to be used to construct general fermionic hamiltonian in dirac operator 
    basis. If rand_H = False, returns non empty arrays A, B; else makes 
    random matrices. 
    
    Blocks of the return define the most general fermionic hamiltonian

    Parameters
    ----------
    dim : int
        dimension of square matrices to be made
    rand_H : bool
        if to make matrices or not
    A : list
        if not random bool, list to be made into matrix A
    B : list
        if not random bool, list to be made into matrix B
    NN: bool
        boolean for seperate construction of nearest neighbor is desired

    Returns
    -------
    As : array
        complex hermetian matrix
    Bs : TYPE
        complex skew symmetric matrix

    """
    if rand_H: 
        if NN == False:
            A = np.random.rand(dim, dim) + 1j * np.random.rand(dim, dim)
            As = 1/2 * (A + cc(A))
            B = np.random.rand(dim, dim) + 1j * np.random.rand(dim, dim)
            for i in range(dim):
                for j in range(dim):
                    if j <= i:
                        B[i,j] = 0
            Bs = B - np.transpose(B)
        else:
            A_v = np.random.rand(3*dim - 2) + 1j * np.random.rand(3*dim - 2) 
            As = np.zeros((dim, dim), dtype = 'complex')
            for k in range(dim):
                As[k,k] = A_v[3*k]
                if k != dim-1:
                    As[k+1,k] = A_v[3*k+1]
                    As[k, k+1] = A_v[3*k+2]
            As = 1/2 * (As + cc(As))
            B_v = np.random.rand(dim-1) + 1j*np.random.rand(dim-1)
            Bs = np.zeros((dim,dim), dtype = 'complex')
            for k in range(dim-1):
                Bs[k,k+1] = B_v[k]
                Bs[k+1,k] = -B_v[k]
    else:
        As = np.array(A)
        Bs = np.array(B)
    return As, Bs

def f_perm(eigs, dim):
    """
    A final permutation matrix is made to re-order the diagonal elements such 
    that diagonal elements are ordered correctly: from most negative 
    to most positive in increasing order
    
    Have to do this to edit scipy's real schur transform. Scipy does not order 
    eigenvalues in a particular way.

    Parameters
    ----------
    eigs : array
        array of positive eigenvalues to be sorted
    dim : int
        number of modes in system

    Returns
    -------
    f_perm : array
        final permutation matrix to be applied to system

    """
    
    order_eigs = np.flip(np.sort(eigs))
    ind_l = []
    for l in range(dim):
        if eigs[l] == order_eigs[l]:
            ind_l.append([l,l])
        else:
            ind_l.append([l, list(order_eigs).index(eigs[l])])
    inds = np.bmat([[np.array(ind_l)], [np.array(ind_l)+dim]])
    perm_f = np.zeros((2*dim, 2*dim))
    for ind in range(2*dim):
        perm_f[inds[ind, 0], inds[ind, 1]] = 1
    f_perm = cc(perm_f)
    return f_perm
    

def Diag(dim, h, om, H):
    """
    Block diagonalizes majorana hamiltonian h with the real schur 
    transform (see Horn and Johnson, section 2.3). Then, constructs 
    permutation matrix so all positive eigenvalues of block are in 
    upper off diagonal of blocks
    
    constructs re-ordering matrix xp_xx to get from Schur, which takes 
    xp ordering, to our computation in which we order xx. Finally, makes
    unitary transform which goes from base particle basis to diagonal 
    quasi-particle basis.

    Parameters
    ----------
    dim : int
        number of modes; returns will be square matrices of 2*dim
    h : array
        random hamiltonian in majorana basis
    om : array
        unitary from base particle basis to majorana basis

    Returns
    -------
    U : TYPE
        DESCRIPTION.
    hdp : TYPE
        DESCRIPTION.

    """
    
    hd, Ot = sp.linalg.schur(h, output = 'real')
    perm = np.zeros((2*dim, 2*dim))
    for j in range(dim):
        if hd[2*j, 2*j + 1] > 0:
            perm[2*j,2*j] = 1
            perm[2*j+1, 2*j+1] = 1
        else:
            perm[2*j,2*j+1] = 1
            perm[2*j+1, 2*j] = 1
    hdp = np.matmul(perm, np.matmul(hd, perm))
    O = np.matmul(Ot, perm)
    xp_xx = np.zeros((2*dim, 2*dim))
    for k in range(dim):
        xp_xx[k, 2*k] = 1
        xp_xx[dim + k, 2*k+1] = 1
    U = cc(np.matmul(np.matmul(cc(om), O), 
                          np.matmul(cc(xp_xx), om)))
    HD = np.real(np.round(np.matmul(U,
                np.matmul(H, cc(U))), decimals = 5))
    fperm = f_perm(np.real(np.diag(HD))[dim:], dim)
    return np.round(np.matmul(fperm, U), decimals = 6), np.round(hdp, decimals = 6), xp_xx

class GFS:
    
    """
    Class for Gaussian Fermionic State
    
    Parameters
    ----------
    dim : int
        number of modes
    As : array
        complex hermetian matrix
    Bs: array
        complex skew symmetric matrix
        
    Attributes
    ----------
    n : int
        number of modes
    H : array
        general fermionic hamiltonian in base particle dirac operator basis
    Omega : array
        unitary transform matrix from base particles to majorana 
    h : array
        hamiltonian in majorana basis
    U : array
        Unitary transform from base particles to diagonal quasiparticle 
        basis: H = U * hd * U^{dagger}
    hd : array
        diagonal hamiltonian in quasiparticle basis
    xp_xx : array
        matrix which re-orders operators into xx order from xp
    Gamma_D : array
        correlation matrix in diagonal basis
    """
    def __init__(self, dim, As, Bs):
        self.n = dim
        self.H = np.bmat([[-np.conjugate(As), Bs],
                          [-np.conjugate(Bs), As]])
        self.Omega= (1 / np.sqrt(2)) * np.bmat([[np.identity(self.n),
                                                 np.identity(self.n)],
                                                [1j* np.identity(self.n),
                                                 -1j * np.identity(self.n)]])
        self.h = np.real(np.matmul(self.Omega, np.matmul(self.H,
                                                         cc(self.Omega)))/1j)
        self.U, self.hd, self.xp_xx = Diag(self.n, self.h, self.Omega, self.H)
        self.Gamma_D = self.Gamma_D_com()
        
    def Gamma_D_com(self):
        """
        Generates diagonal correlation matrix using rho values 

        Returns
        -------
        Gamma : array
            diagonal correlation matrix

        """
        eigs = np.real(np.diag(self.Diagonal_H()))[self.n:]
        Gamma_D = np.zeros((2*self.n, 2*self.n))
        for j in range(self.n):
            Gamma_D[j, j] = 1/2 * (1 + np.tanh(eigs[j]))
            Gamma_D[j+self.n, j+self.n] = 1/2 * (1 - np.tanh(eigs[j]))
        return Gamma_D
        
    def Diagonal_H(self):
        """
        returns diagonal of dirac matrix H

        Returns
        -------
        HD : array
            diagonal form of H, from U H U^{dagger}

        """
        HD = np.real(np.round(np.matmul(self.U,
                      np.matmul(self.H, cc(self.U))), decimals = 5))
        return HD
    
    def Gamma_H_com(self):
        """
        returns correlation matrix in original dirac operator basis

        Returns
        -------
        Gamma_H : array
            correlation matrix in dirac operator basis, 
            Gamma_H = U^{dagger} Gamma_D U^{dagger}

        """
        Gamma_H = np.matmul(cc(self.U), np.matmul(self.Gamma_D, self.U))
        return Gamma_H
    
    def Ground_State(self):
        """
        Returns correlation matrix of ground state in dirac basis
        
        ground state in diagonal basis has no occupied quasi-modes, so 
        diagonal gamma has bottom left block as identity 

        Returns
        -------
        Gamma_0 : array
            correlation matrix of ground state 

        """
        Gamma_0D = np.bmat[[np.zeros((self.n, self.n)), np.zeros((self.n, self.n))],
                           [np.zeros((self.n, self.n)), np.identity(self.n)]]
        Gamma_0 = np.matmul(cc(self.U), np.matmul(Gamma_0D, self.U))
        return Gamma_0
    
    def Energy_G(self):
        """
        Returns total energy 

        Returns
        -------
        E : float
            Total energy for rho defined on specific hamiltonian H

        """
        E = 0
        eigs = np.real(np.diag(self.Diagonal_H()))[self.n:]
        for j in range(self.n):
            E += eigs[j]*(self.Gamma_D[j,j] - self.Gamma_D[j + self.n,j+self.n])
        return E
    
    def Eig_of_Rho(self):
        """
        Returns list of eigenvalues of diagonal density matrix in {1,0}^N 
        ordered basis

        Returns
        -------
        eig : array
            array of values representing diagonal of density matrix

        """
        eigs = np.real(np.diag(self.Diagonal_H()))[self.n:]
        eig = np.array(eigs[self.n-1], 1 - eigs[self.n-1])
        for j in range(self.n - 1):
            eig = np.bmat([eigs[self.n-2-j]*eig, (1-eigs[self.n-2-j])*eig])
        return eig
 
    def Reduce_Gamma_H_com(self, mode_l):
        """
        Reduce's gamma by partial tracing over list of modes, deleting rows
        and columns from Gamma accordingly 

        Arguements
        ----------
        mode_l : list
            list of indices to reduce over
        
        Returns
        -------
        Gamma_H : array
            array of reduced gamma 

        """
        
        modes = mode_l
        t_modes = np.arange(0, self.n, 1)
        k_modes = np.setdiff1d(t_modes, modes)
        k_mode = np.append(k_modes, k_modes+self.n)
        Gamma_H = self.Gamma_H_com()
        Gamma_RR = Gamma_H[k_mode]
        Gamma_RCR = Gamma_RR[:, k_mode]
        return np.reshape(Gamma_RCR, (len(k_mode), len(k_mode)))

    def VN_entropy(self, Gamma):
        """
        Calculates Von Neumann Entropy of g.f.s given gamma 

        Returns
        -------
        S : float
            value of von neumann entropy 

        """
        l = len(Gamma)
        S = 0
        for j in np.arange(l):
            S += -1 * (Gamma[j,j]*np.log(Gamma[j,j]))
        return S
    
    def Purity(self):
        """
        Returns purity measure of entanglement from state        
        
        Returns
        -------
        p : float
            Purity of f.g.s from energy eigenvalues

        """
        
        p = 1
        eigs = np.real(np.diag(self.Diagonal_H()))[self.n:]
        for en in eigs:
            p = p * 1/(1/np.cosh(en) + 1)
        return p
    
    def Fourier_W(self):
        """
        creates block fourier matrix to diagonalize correlation matrices 
        for translationally invariant states

        Returns
        -------
        F_mat : array
            block matrix form of fourier matrix 

        """
        fmat = dft(self.n)
        fmat[[0,self.n-1]] = fmat[[self.n-1, 0]]
        fmat[:,[0,self.n-1]] = fmat[:,[self.n-1, 0]]
        F_mat = np.bmat()[[fmat, np.identity(self.n)],
                          [np.identity(self.n), np.conjugate(fmat)]]
        return F_mat
    
    def Entanglement_Contour(self, mode_l):
        """
        Calculate the entanglement contour for a system over a given list of 
        nodes. 

        Parameters
        ----------
        mode_l : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        modes = np.setdiff1d(np.arange(0, self.n, 1), mode_l)
        c_sites = np.zeros(len(modes))
        for j in range(len(modes)):
            ind = modes[j]
            for k in modes:
                p_j_k = 1/2 * (an(self.U[ind,k]) + an(self.U[ind +self.n, k+self.n]) 
                               +an(self.U[ind, k+self.n]) + an(self.U[ind+self.n, k]))
                vk = self.Gamma_D[k,k]
                c_sites[j] += p_j_k * (vk*np.log(vk) + (1-vk)*np.log(1-vk))   
        return c_sites
                
#%%   
"""
dim = 8
As, Bs = init_mats(dim, rand_H = True, A = [], B = [], NN = True)
GFSt = GFS(dim, As, Bs)
eigs = np.real(np.diag(GFSt.Diagonal_H()))[:GFSt.n]
arr = np.array([2,3,4,5])
Gamma_R = GFSt.Reduce_Gamma_H_com(arr)
"""
#%%
### Epsilon values for NN Hamiltonian 
dim = 64
As, Bs = init_mats(dim, rand_H = True, A = [], B = [], NN = True)
GFSE = GFS(dim, As, Bs)

plt.plot(np.arange(0,dim,1),np.real(np.diag(GFSE.Diagonal_H()))[:dim])
plt.show()

arr = np.arange(17,48,1)
Gamma_R = GFSE.Reduce_Gamma_H_com(arr)
vn = GFSE.VN_entropy(Gamma_R)