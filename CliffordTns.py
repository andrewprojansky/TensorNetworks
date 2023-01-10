#%%
"""
Code for understanding clifford simulation with tensor networks

Project Start Date: 9/21/22
"""

import numpy as np
import random
import time
import matplotlib.pyplot as plt

"""
Defines basis set of clifford gates, and all useful arrays/dictionarys for 
later simulation
"""

H = np.array([[1 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), -1 / np.sqrt(2)]])
S = np.array([[1, 0], [0, 1j]])
CNotF = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
CNotB = np.array([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]])
Id = np.array([[1, 0], [0, 1]])
zero = np.array([0,0])
#rho_0 = np.tensordot(zero, zero,0)
gate_dict = {1: S, 2: H, 3: "CNOT"}

def cc(gate):
    """
    Parameters
    ----------
    gate : array
        input gate to be applied to state

    Returns
    -------
    array
        complex conjuagte transpose of gate

    """

    return np.transpose(np.matrix.conjugate(gate))

def initial_state(dim, bra):
    """
    
    Parameters
    ----------
    dim : int
        2 to the number of qubits

    Returns
    -------
    state : array
        array/matrix of all zeros besides single eigenstate entry

    """
    
    if bra == False:
        state = np.zeros((2**dim,2**dim))
        state[0][0] = 1
    else:
        state = np.zeros(2**dim)
        state[0] = 1
    return state

class Experiment:
    """
    Can either run experiment creating clifford circuits by evolving density 
    matrix via gate application, or uses the stabalizer framework to evolve
    state by evolving the operators
    
    Parameters
    ----------
    dim: int
        number of qubits
    state: list, optional
        Initial state position. The default is the plus Z eigenstate
    num_steps : int, optional
        total number of steps. The default is 1.
    gd: dict, optional
        set of gates for non stabalizer formalism gate simulation
    stabalizer: bool, optional
        boolean as to whether run random gate circuit or random stabalizer circuit
    sgd: dict, optional
        list of gates for stabalizer circuit simulation
    
    """

    def __init__(
        self,
        dim,
        bra,
        num_steps: int = 1,
        gd = gate_dict,

    ):
        self.num_steps = num_steps
        self.dim = dim
        self.state = initial_state(dim, bra)
        self.bra = bra
        self.gd = gd
        self.time_r = 0
        
    ################# non-stabalizer gate functions ##################################
    """
    creates gates of correct size to multiply density operator
    """
    
    def OneQGate(self, gate, qubit):
        """
        Parameters
        ----------
        gate : int (keyword)
            keyword for base gate access from dictionary
        qubit : int
            qubit gate applied to

        Returns
        -------
        fgate : array
            returns gate of correct dimension for Hilbert space

        """

        if qubit == 0:
            fgate = gate
        else:
            fgate = Id
        for i in range(1, self.dim):
            if qubit == i:
                fgate = np.kron(gate, fgate)
            else:
                fgate = np.kron(Id, fgate)
        return fgate

    def CNOT(self, control, target):
        """
        Parameters
        ----------
        control : int
            arbitrary control qubit for CNot application
        target : int
            arbitrary target qubit for CNot application

        Returns
        -------
        fgate : array
            returns cnot gate of correct dimension for Hilbert space


        """
        
        if target > control:
            if target - 1 == 0: 
                fgate = CNotF
            else: 
                fgate = Id
            for i in range(1, self.dim - 1):
                if i == target-1:
                    fgate = np.kron(CNotF, fgate)
                else:
                    fgate = np.kron(Id, fgate)
            for i in range(target - 1, control, -1):
                if i-1 == 0:
                    sgate = CNotF
                else:
                    sgate = Id
                for j in range(1, self.dim-1):
                    if j == i-1:
                        sgate= np.kron(CNotF, sgate)
                    else:
                        sgate = np.kron(Id, sgate)                
                fgate = np.matmul(sgate, fgate)
                fgate = np.matmul(fgate, fgate)
        elif target < control: 
            if control == 1:
                fgate = CNotB
            else:
                fgate = Id
            for i in range(1, self.dim-1):
                if i + 1 == control:
                    fgate = np.kron(CNotB, fgate)
                else:
                    fgate = np.kron(Id, fgate)
            for i in range(target, control-1):
                if i == 0: 
                    sgate = CNotB
                else:
                    sgate = Id
                for j in range(1, self.dim-1):
                    if j == i: 
                        sgate = np.kron(CNotB, sgate)
                    else:
                        sgate = np.kron(Id, sgate)
                fgate = np.matmul(sgate, fgate)
                fgate = np.matmul(fgate, fgate)
        return fgate
                
    ################# run functions ##################################
    def experiment(self):
        """
        Depending on if a stabalizer circuit or not, applies number of
        randomly chosen gates to state of system

        """
        
        start = time.time()
        for i in range(self.num_steps):
            gate = random.randint(1, len(self.gd))
            if self.gd[gate] == "CNOT":
                control = random.randint(0, self.dim-1)
                target = random.randint(0, self.dim-1)
                while control == target: 
                    target = random.randint(0, self.dim-1)
                Cgate = self.CNOT(control, target)

            else:
                qubit = random.randint(0, self.dim-1)
                Cgate = self.OneQGate(self.gd[gate], qubit)
                
            if self.bra == False:    
                self.state = np.matmul(Cgate, np.matmul(self.state, cc(Cgate)))
            else:
                self.state = np.matmul(Cgate, self.state)
        end = time.time()
        self.time_r = end-start
        
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
        
        Psi = np.zeros((2,mdim//2), dtype='complex')
        for i in range(mdim//2):
            Psi[0,i] = self.state[i]
            Psi[1,i] = self.state[mdim//2+i]
        return Psi
    

    def Make_TN(self):
        '''
        Makes TN for each site, getting A-sigma matrices at each site with left 
        normalization at all except for last
        '''
        
        Psi = self.Psi_1(2**self.dim)
        prank = 1
        for i in range(dim-1):
            u, s, vh = np.linalg.svd(Psi, full_matrices=True)
            s = np.around(s, decimals = 10)
            s = s[0:len(np.nonzero(s)[0])]
            rank = len(s)
            u = u[:, :rank]
            u = u.reshape((2,prank,rank))
            u = np.round(u, 10)
            self.TNW[i] = u
            vh = vh[:rank,:]
            Psi = np.reshape(np.matmul(np.diag(s), vh), (rank*2, 2**(dim-2-i)))
            prank = rank
        prank = 1
        Psi = Psi.reshape(2, rank, prank)
        Psi = np.round(Psi, 10)
        self.TNW[dim-1] = Psi
    
    def Multiplication(self, bi: str):
        if len(self.TNW.keys()) != len(bi):
            print('Need new string')
            return
        else:
            return np.linalg.multi_dot([x[int(bi[i])] for i,x in enumerate(self.TNW.values())])
#%%
l = []
for j in range(1000):
    start = time.time()
    m_l = []
    dim = 6
    exp = Experiment(dim, bra = True, num_steps=dim**2)
    exp.experiment()
    Tn = TensorNetwork(dim, exp.state)
    Tn.Make_TN()
    for key in Tn.TNW:
        m_l.append(max(np.shape(Tn.TNW[key])))
    l.append(max(m_l))
    end = time.time()
    print(end-start)
 
plt.hist(np.log2(l), bins=dim//2)
    