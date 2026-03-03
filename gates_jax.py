#Define the various gates 
import jax
import numpy as np
import jax.numpy as jnp
import scipy.sparse as sp
from scipy.linalg import null_space
import time
from itertools import product
import functools as ft


def print_dm(matrix):
    """Prints a 2D matrix in a readable, tab-separated format with each element
    formatted to three decimal places.

    Args:
        matrix (list of lists): A 2D list (matrix) of numerical values to be printed.
    """
    for row in matrix:
        print("[", end="")
        print(*[f"{entry:.3f}" for entry in row], sep="\t", end="")
        print("]")

def convert_to_base_four(i):
    """Converts a number in the range [0, 16] to base 4.
    Args:
        i (int): A number in the range [0, 16]. 

    Returns: 
        tuple: The first and second bits in base 4.
    """
    first_bit = i//4
    second_bit = i%4
    return first_bit, second_bit

def expand_gate_Bloch(M, qubits, N_qubits):
    """Expands a gate matrix to act on a specific pair of qubits in a multi-qubit system.

    Args:    
        M: 16x16 matrix that transforms the bloch vector of two qubits
        qubits: list of two integers specifying the location of the two qubits the gate acts on
        N_qubits: total number of qubits in the circuit  

    Returns   
        the 4^N x 4^N matrix that transforms the bloch vector of the whole system.  
    """
    l,m = qubits
    dim = 4 ** N_qubits
    expanded_matrix = jnp.zeros((dim, dim), dtype=float)
    # Update the expanded matrix
    # Goes through entries of the matrix and adds 
    for i in range(16):
        for j in range(16):
            #Check if l is smaller than m if not reverse order
            if l<m: 
                i1,i2 = convert_to_base_four(i)
                j1,j2 = convert_to_base_four(j)
            elif l>m:
                #print("l>m")
                i1,i2 = convert_to_base_four(i)
                j1,j2 = convert_to_base_four(j)
                #i2,i1 = convert_to_base_four(i)
                #j2,j1 = convert_to_base_four(j)
                column_order = ['II', 'IX', 'IY', 'IZ', 'XI', 'XX', 'XY', 'XZ', 'YI', 'YX', 'YY', 'YZ', 'ZI', 'ZX', 'ZY', 'ZZ']
                trsp_column_order = np.transpose(np.reshape(np.array(column_order),(4,4))).flatten()
                new_column_mapping = {col: i for i, col in enumerate(trsp_column_order)}
                M = M[:, [new_column_mapping[col] for col in column_order]]
                M = M[ [new_column_mapping[col] for col in column_order],:]
                
                m,l = l,m
            else:
                raise ValueError("the indices must be distinct")
            
            #Pad every entry with identies inbetween
            if l != 0:
                P1 = M[i, j] * jnp.kron(jnp.eye(4**l), jnp.array(sp.csr_array(([1], ([i1], [j1])), shape=(4, 4)).toarray()))

            else: 
                P1 = M[i,j] *jnp.array(sp.csr_array(([1], ([i1],[j1])), shape=(4, 4)).toarray())

            if int(m-l) != 1:
                P2 = jnp.kron(jnp.kron(P1,jnp.eye(4**(np.abs(m-l-1)))),jnp.array(sp.csr_array(([1], ([i2],[j2])), shape=(4, 4)).toarray()))

            else: 
                P2 = jnp.kron(P1,jnp.array(sp.csr_array(([1], ([i2],[j2])), shape=(4, 4)).toarray()))

            if int(N_qubits-m-1) > 0:
                P3 = jnp.kron(P2,jnp.eye(4**(N_qubits-m-1)))
            else: 
                P3 = P2
            expanded_matrix += P3
    return expanded_matrix

def kron_multi(op,m): 
    """Computes the kron product of an operator with itself m times.

    Args:
        op (ndarray): The operator to multiply.
        m (int): The number of times to multiply the operator.

    Returns:
        ndarray: The Kronecker product of the operator with itself m times.

    """
    if not isinstance(m, int):
        raise ValueError("Value must be an integer. Got "+str(type(m))+" instead.")
    product = op
    for i in range(m-1):
        product = jnp.kron(product,op)
    return product


class Gate:
    """A class representing a quantum gate.

    TODO: 
        rewrite using regiestry to make it cleaner

    Attributes:
        name (str): The name of the gate.
        params (list): The parameters of the gate.
        N_qubits (int): The total number of qubits.
        Gate_loc (int or list): The location of the gate.
        pattern (str): The pattern for broadcasting the gate.
    """
    def __init__(self, name,params = None, N_qubits = None, Gate_loc = None,pattern = None):
        """Initializes a Gate object.

        Args:
            name (str): The name of the gate.
            params (list, optional): The parameters of the gate.
            N_qubits (int, optional): The total number of qubits.
            Gate_loc (int or list, optional): The location of the gate.
            pattern (str, optional): The pattern for broadcasting the gate.
        """
        self.name = name
        self.params = params
        if N_qubits:
            self.N_qubits = N_qubits
        else: self.N_qubits = None
        if Gate_loc is not None:
            self.Gate_loc = Gate_loc
        else: self.Gate_loc = None
        self.pattern = pattern

    def n_params(self):
        """Returns the number of free parameters of a gate.

        Returns:
            int: The number of free parameters.
        """
        if self.name == "RX" or self.name == "RZ" or self.name == "RY" or self.name == "RY_Z":
            return 1
        elif self.name == "RXX" or self.name == "RYY" or self.name == "RZZ" or self.name == "IsingXX" or self.name == "IsingYY" or self.name == "IsingZZ" or self.name =="XXPlusYY" or self.name =="XXMinusYY":
            return 1
        elif self.name == "CRX" or self.name == "CRY" or self.name == "CRZ" or self.name == "CRX_S":
            return 1
        elif self.name == "AmplitudeDamping":
            return 1
        elif self.name == "Depolarizing":
            return 1
        else: return 0
    
    def _get_matrix(self):
        """Returns the matrix that transforms the coherence vector.

        Returns:
            ndarray: The matrix that transforms the coherence vector.
        """
        ###########################################################
        ############### Single qubit non parameterized ############
        ###########################################################

        if self.name == 'H' or self.name == "Hadamard":
            return jnp.array( [jnp.array( [1,0,0,0,] ),jnp.array( [0,0,0,1,] ),jnp.array( [0,0,-1,0,] ),jnp.array( [0,1,0,0,] ),] )
        
        elif self.name == 'X' or self.name == 'PauliX':
            return jnp.array( [jnp.array( [1,0,0,0,] ),jnp.array( [0,1,0,0,] ),jnp.array( [0,0,-1,0,] ),jnp.array( [0,0,0,-1,] ),] )
        
        elif self.name == 'Y' or self.name == 'PauliY': 
            return jnp.array( [jnp.array( [1,0,0,0,] ),jnp.array( [0,-1,0,0,] ),jnp.array( [0,0,1,0,] ),jnp.array( [0,0,0,-1,] ),] )
        
        elif self.name == 'Z' or self.name == 'PauliZ':
            return jnp.array( [jnp.array( [1,0,0,0,] ),jnp.array( [0,-1,0,0,] ),jnp.array( [0,0,-1,0,] ),jnp.array( [0,0,0,1,] ),] )
        
        elif self.name == 'T':
            return jnp.array( [jnp.array( [1,0,0,0,] ),jnp.array( [0,( 2 )**( -1/2 ),-1 * ( 2 )**( -1/2 ),0,] ),jnp.array( [0,( 2 )**( -1/2 ),( 2 )**( -1/2 ),0,] ),jnp.array( [0,0,0,1,] ),] )
        
        elif self.name == "S":
            return jnp.array( [jnp.array( [1,0,0,0,] ),jnp.array( [0,0,-1,0,] ),jnp.array( [0,1,0,0,] ),jnp.array( [0,0,0,1,] ),] )
        
        ###########################################################
        ############### Single qubit parameterized ################
        ###########################################################
        elif self.name == "RX":
            if self.params is None:
                raise ValueError("Parameter 'params' must be provided for RX gate.")
            phi = self.params[0]
            return jnp.array( [jnp.array( [jnp.cosh( jnp.imag( phi ) ),jnp.sinh( jnp.imag( phi ) ),0,0,] ),jnp.array( [jnp.sinh( jnp.imag( phi ) ),jnp.cosh( jnp.imag( phi ) ),0,0,] ),jnp.array( [0,0,jnp.cos( jnp.real( phi ) ),-1 * jnp.sin( jnp.real( phi ) ),] ),jnp.array( [0,0,jnp.sin( jnp.real( phi ) ),jnp.cos( jnp.real( phi ) ),] ),] )
        
        elif self.name == "RY":
            if self.params is None:
                raise ValueError("Parameter 'params' must be provided for RY gate.")
            phi =  self.params[0]
            return jnp.array( [jnp.array( [jnp.cosh( jnp.imag( phi ) ),0,jnp.sinh( jnp.imag( phi ) ),0,] ),jnp.array( [0,jnp.cos( jnp.real( phi ) ),0,jnp.sin( jnp.real( phi ) ),] ),jnp.array( [jnp.sinh( jnp.imag( phi ) ),0,jnp.cosh( jnp.imag( phi ) ),0,] ),jnp.array( [0,-1 * jnp.sin( jnp.real( phi ) ),0,jnp.cos( jnp.real( phi ) ),] ),] )
        
        elif self.name == "RZ":
            if self.params is None:
                raise ValueError("Parameter 'params' must be provided for RZ gate.")
            phi =  self.params[0]
            return jnp.array( [jnp.array( [1/2 * ( jnp.e )**( complex( 0,-1 ) * jnp.real( phi ) ) * ( ( jnp.e )**( complex( 0,1 ) * phi ) + ( jnp.e )**( complex( 0,1 ) * jnp.conjugate( phi ) ) ),0,0,1/2 * ( jnp.e )**( complex( 0,-1 ) * jnp.real( phi ) ) * ( -1 * ( jnp.e )**( complex( 0,1 ) * phi ) + ( jnp.e )**( complex( 0,1 ) * jnp.conjugate( phi ) ) ),] ),jnp.array( [0,jnp.cos( jnp.real( phi ) ),-1 * jnp.sin( jnp.real( phi ) ),0,] ),jnp.array( [0,jnp.sin( jnp.real( phi ) ),jnp.cos( jnp.real( phi ) ),0,] ),jnp.array( [1/2 * ( jnp.e )**( complex( 0,-1 ) * jnp.real( phi ) ) * ( -1 * ( jnp.e )**( complex( 0,1 ) * phi ) + ( jnp.e )**( complex( 0,1 ) * jnp.conjugate( phi ) ) ),0,0,1/2 * ( jnp.e )**( complex( 0,-1 ) * jnp.real( phi ) ) * ( ( jnp.e )**( complex( 0,1 ) * phi ) + ( jnp.e )**( complex( 0,1 ) * jnp.conjugate( phi ) ) ),] ),] )
        elif self.name == "RY_Z":
        #RY gate followed by a S gate, for param = Pi/2 implememts a Hadamard gate
            if self.params is None:
                raise ValueError("Parameter 'params' must be provided for RZ gate.")
            phi =  self.params[0]
            return jnp.array( [jnp.array( [jnp.cosh( jnp.imag( phi ) ),0,jnp.sinh( jnp.imag( phi ) ),0,] ),jnp.array( [0,-1 * jnp.cos( jnp.real( phi ) ),0,-1 * jnp.sin( jnp.real( phi ) ),] ),jnp.array( [-1 * jnp.sinh( jnp.imag( phi ) ),0,-1 * jnp.cosh( jnp.imag( phi ) ),0,] ),jnp.array( [0,-1 * jnp.sin( jnp.real( phi ) ),0,jnp.cos( jnp.real( phi ) ),] ),] )
        
        ###########################################################
        ############### Two qubit non parameterized ###############
        ###########################################################

        elif self.name == 'XI':
            #X otimes I for test porposes
            return jnp.array( [jnp.array( [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,] ),jnp.array( [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,] ),jnp.array( [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,] ),jnp.array( [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,] ),jnp.array( [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,] ),jnp.array( [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,] ),jnp.array( [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,] ),jnp.array( [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,] ),jnp.array( [0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,] ),jnp.array( [0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,] ),jnp.array( [0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,] ),jnp.array( [0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,] ),jnp.array( [0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,] ),jnp.array( [0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,] ),jnp.array( [0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,] ),jnp.array( [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,] ),] )
        
        elif self.name == "CNOT":
            return jnp.array( [jnp.array( [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,] ),jnp.array( [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,] ),jnp.array( [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,] ),jnp.array( [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,] ),jnp.array( [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,] ),jnp.array( [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,] ),jnp.array( [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,] ),jnp.array( [0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,] ),jnp.array( [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,] ),jnp.array( [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,] ),jnp.array( [0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,] ),jnp.array( [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,] ),jnp.array( [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,] ),jnp.array( [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,] ),jnp.array( [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,] ),jnp.array( [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,] ),] )
        
        elif self.name == "SWAP":
            return jnp.array( [jnp.array( [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,] ),jnp.array( [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,] ),jnp.array( [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,] ),jnp.array( [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,] ),jnp.array( [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,] ),jnp.array( [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,] ),jnp.array( [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,] ),jnp.array( [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,] ),jnp.array( [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,] ),jnp.array( [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,] ),jnp.array( [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,] ),jnp.array( [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,] ),jnp.array( [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,] ),jnp.array( [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,] ),jnp.array( [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,] ),jnp.array( [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,] ),] )
        
        ###########################################################
        ############### Two qubit parameterized ###################
        ###########################################################

        elif self.name == "RXX" or self.name == "IsingXX":
            if self.params is None:
                raise ValueError("Parameter 'params' must be provided for RXX gate.")
            t =  self.params[0]
            return jnp.array( [jnp.array( [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,] ),jnp.array( [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,] ),jnp.array( [0,0,jnp.cos( t ),0,0,0,0,-1 * jnp.sin( t ),0,0,0,0,0,0,0,0,] ),jnp.array( [0,0,0,jnp.cos( t ),0,0,jnp.sin( t ),0,0,0,0,0,0,0,0,0,] ),jnp.array( [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,] ),jnp.array( [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,] ),jnp.array( [0,0,0,-1 * jnp.sin( t ),0,0,jnp.cos( t ),0,0,0,0,0,0,0,0,0,] ),jnp.array( [0,0,jnp.sin( t ),0,0,0,0,jnp.cos( t ),0,0,0,0,0,0,0,0,] ),jnp.array( [0,0,0,0,0,0,0,0,jnp.cos( t ),0,0,0,0,-1 * jnp.sin( t ),0,0,] ),jnp.array( [0,0,0,0,0,0,0,0,0,jnp.cos( t ),0,0,-1 * jnp.sin( t ),0,0,0,] ),jnp.array( [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,] ),jnp.array( [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,] ),jnp.array( [0,0,0,0,0,0,0,0,0,jnp.sin( t ),0,0,jnp.cos( t ),0,0,0,] ),jnp.array( [0,0,0,0,0,0,0,0,jnp.sin( t ),0,0,0,0,jnp.cos( t ),0,0,] ),jnp.array( [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,] ),jnp.array( [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,] ),] )
        
        elif self.name == "RYY" or self.name == "IsingYY":
            if self.params is None:
                raise ValueError("Parameter 'params' must be provided for RYY gate.")
            t =  self.params[0]
            return jnp.array( [jnp.array( [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,] ),jnp.array( [0,jnp.cos( t ),0,0,0,0,0,0,0,0,0,jnp.sin( t ),0,0,0,0,] ),jnp.array( [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,] ),jnp.array( [0,0,0,jnp.cos( t ),0,0,0,0,0,-1 * jnp.sin( t ),0,0,0,0,0,0,] ),jnp.array( [0,0,0,0,jnp.cos( t ),0,0,0,0,0,0,0,0,0,jnp.sin( t ),0,] ),jnp.array( [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,] ),jnp.array( [0,0,0,0,0,0,jnp.cos( t ),0,0,0,0,0,jnp.sin( t ),0,0,0,] ),jnp.array( [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,] ),jnp.array( [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,] ),jnp.array( [0,0,0,jnp.sin( t ),0,0,0,0,0,jnp.cos( t ),0,0,0,0,0,0,] ),jnp.array( [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,] ),jnp.array( [0,-1 * jnp.sin( t ),0,0,0,0,0,0,0,0,0,jnp.cos( t ),0,0,0,0,] ),jnp.array( [0,0,0,0,0,0,-1 * jnp.sin( t ),0,0,0,0,0,jnp.cos( t ),0,0,0,] ),jnp.array( [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,] ),jnp.array( [0,0,0,0,-1 * jnp.sin( t ),0,0,0,0,0,0,0,0,0,jnp.cos( t ),0,] ),jnp.array( [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,] ),] )
        
        elif self.name == "RZZ" or self.name == "IsingZZ":
            if self.params is None:
                raise ValueError("Parameter 'params' must be provided for RZZ gate.")
            t =  self.params[0]
            return jnp.array( [jnp.array( [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,] ),jnp.array( [0,jnp.cos( t ),0,0,0,0,0,0,0,0,0,0,0,0,-1 * jnp.sin( t ),0,] ),jnp.array( [0,0,jnp.cos( t ),0,0,0,0,0,0,0,0,0,0,jnp.sin( t ),0,0,] ),jnp.array( [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,] ),jnp.array( [0,0,0,0,jnp.cos( t ),0,0,0,0,0,0,-1 * jnp.sin( t ),0,0,0,0,] ),jnp.array( [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,] ),jnp.array( [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,] ),jnp.array( [0,0,0,0,0,0,0,jnp.cos( t ),-1 * jnp.sin( t ),0,0,0,0,0,0,0,] ),jnp.array( [0,0,0,0,0,0,0,jnp.sin( t ),jnp.cos( t ),0,0,0,0,0,0,0,] ),jnp.array( [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,] ),jnp.array( [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,] ),jnp.array( [0,0,0,0,jnp.sin( t ),0,0,0,0,0,0,jnp.cos( t ),0,0,0,0,] ),jnp.array( [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,] ),jnp.array( [0,0,-1 * jnp.sin( t ),0,0,0,0,0,0,0,0,0,0,jnp.cos( t ),0,0,] ),jnp.array( [0,jnp.sin( t ),0,0,0,0,0,0,0,0,0,0,0,0,jnp.cos( t ),0,] ),jnp.array( [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,] ),] )
        
        elif self.name == "XXPlusYY":
            if self.params is None:
                raise ValueError("Parameter 'params' must be provided for XXPlusYY gate.")
            t =  self.params[0]
            return jnp.array( [jnp.array( [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,] ),jnp.array( [0,jnp.cos( t ),0,0,0,0,0,0,0,0,0,jnp.sin( t ),0,0,0,0,] ),jnp.array( [0,0,jnp.cos( t ),0,0,0,0,-1 * jnp.sin( t ),0,0,0,0,0,0,0,0,] ),jnp.array( [0,0,0,( jnp.cos( t ) )**( 2 ),0,0,jnp.cos( t ) * jnp.sin( t ),0,0,-1 * jnp.cos( t ) * jnp.sin( t ),0,0,( jnp.sin( t ) )**( 2 ),0,0,0,] ),jnp.array( [0,0,0,0,jnp.cos( t ),0,0,0,0,0,0,0,0,0,jnp.sin( t ),0,] ),jnp.array( [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,] ),jnp.array( [0,0,0,-1 * jnp.cos( t ) * jnp.sin( t ),0,0,( jnp.cos( t ) )**( 2 ),0,0,( jnp.sin( t ) )**( 2 ),0,0,jnp.cos( t ) * jnp.sin( t ),0,0,0,] ),jnp.array( [0,0,jnp.sin( t ),0,0,0,0,jnp.cos( t ),0,0,0,0,0,0,0,0,] ),jnp.array( [0,0,0,0,0,0,0,0,jnp.cos( t ),0,0,0,0,-1 * jnp.sin( t ),0,0,] ),jnp.array( [0,0,0,jnp.cos( t ) * jnp.sin( t ),0,0,( jnp.sin( t ) )**( 2 ),0,0,( jnp.cos( t ) )**( 2 ),0,0,-1 * jnp.cos( t ) * jnp.sin( t ),0,0,0,] ),jnp.array( [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,] ),jnp.array( [0,-1 * jnp.sin( t ),0,0,0,0,0,0,0,0,0,jnp.cos( t ),0,0,0,0,] ),jnp.array( [0,0,0,( jnp.sin( t ) )**( 2 ),0,0,-1 * jnp.cos( t ) * jnp.sin( t ),0,0,jnp.cos( t ) * jnp.sin( t ),0,0,( jnp.cos( t ) )**( 2 ),0,0,0,] ),jnp.array( [0,0,0,0,0,0,0,0,jnp.sin( t ),0,0,0,0,jnp.cos( t ),0,0,] ),jnp.array( [0,0,0,0,-1 * jnp.sin( t ),0,0,0,0,0,0,0,0,0,jnp.cos( t ),0,] ),jnp.array( [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,] ),] )
        
        elif self.name == "XXMinusYY":
            if self.params is None:
                raise ValueError("Parameter 'params' must be provided for XXMinusYY gate.")
            t =  self.params[0]
            return jnp.array( [jnp.array( [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,] ),jnp.array( [0,jnp.cos( t ),0,0,0,0,0,0,0,0,0,-1 * jnp.sin( t ),0,0,0,0,] ),jnp.array( [0,0,jnp.cos( t ),0,0,0,0,-1 * jnp.sin( t ),0,0,0,0,0,0,0,0,] ),jnp.array( [0,0,0,( jnp.cos( t ) )**( 2 ),0,0,jnp.cos( t ) * jnp.sin( t ),0,0,jnp.cos( t ) * jnp.sin( t ),0,0,-1 * ( jnp.sin( t ) )**( 2 ),0,0,0,] ),jnp.array( [0,0,0,0,jnp.cos( t ),0,0,0,0,0,0,0,0,0,-1 * jnp.sin( t ),0,] ),jnp.array( [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,] ),jnp.array( [0,0,0,-1 * jnp.cos( t ) * jnp.sin( t ),0,0,( jnp.cos( t ) )**( 2 ),0,0,-1 * ( jnp.sin( t ) )**( 2 ),0,0,-1 * jnp.cos( t ) * jnp.sin( t ),0,0,0,] ),jnp.array( [0,0,jnp.sin( t ),0,0,0,0,jnp.cos( t ),0,0,0,0,0,0,0,0,] ),jnp.array( [0,0,0,0,0,0,0,0,jnp.cos( t ),0,0,0,0,-1 * jnp.sin( t ),0,0,] ),jnp.array( [0,0,0,-1 * jnp.cos( t ) * jnp.sin( t ),0,0,-1 * ( jnp.sin( t ) )**( 2 ),0,0,( jnp.cos( t ) )**( 2 ),0,0,-1 * jnp.cos( t ) * jnp.sin( t ),0,0,0,] ),jnp.array( [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,] ),jnp.array( [0,jnp.sin( t ),0,0,0,0,0,0,0,0,0,jnp.cos( t ),0,0,0,0,] ),jnp.array( [0,0,0,-1 * ( jnp.sin( t ) )**( 2 ),0,0,jnp.cos( t ) * jnp.sin( t ),0,0,jnp.cos( t ) * jnp.sin( t ),0,0,( jnp.cos( t ) )**( 2 ),0,0,0,] ),jnp.array( [0,0,0,0,0,0,0,0,jnp.sin( t ),0,0,0,0,jnp.cos( t ),0,0,] ),jnp.array( [0,0,0,0,jnp.sin( t ),0,0,0,0,0,0,0,0,0,jnp.cos( t ),0,] ),jnp.array( [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,] ),] )
        
        elif self.name == "CRX":
            if self.params is None:
                raise ValueError("Parameter 'params' must be provided for CRX gate.")
            phi =  self.params[0]
            return jnp.array( [jnp.array( [( jnp.cosh( 1/2 * jnp.imag( phi ) ) )**( 2 ),1/2 * jnp.sinh( jnp.imag( phi ) ),0,0,0,0,0,0,0,0,0,0,-1 * ( jnp.sinh( 1/2 * jnp.imag( phi ) ) )**( 2 ),-1/2 * jnp.sinh( jnp.imag( phi ) ),0,0,] ),jnp.array( [1/2 * jnp.sinh( jnp.imag( phi ) ),( jnp.cosh( 1/2 * jnp.imag( phi ) ) )**( 2 ),0,0,0,0,0,0,0,0,0,0,-1/2 * jnp.sinh( jnp.imag( phi ) ),-1 * ( jnp.sinh( 1/2 * jnp.imag( phi ) ) )**( 2 ),0,0,] ),jnp.array( [0,0,( jnp.cos( 1/2 * jnp.real( phi ) ) )**( 2 ),-1/2 * jnp.sin( jnp.real( phi ) ),0,0,0,0,0,0,0,0,0,0,( jnp.sin( 1/2 * jnp.real( phi ) ) )**( 2 ),1/2 * jnp.sin( jnp.real( phi ) ),] ),jnp.array( [0,0,1/2 * jnp.sin( jnp.real( phi ) ),( jnp.cos( 1/2 * jnp.real( phi ) ) )**( 2 ),0,0,0,0,0,0,0,0,0,0,-1/2 * jnp.sin( jnp.real( phi ) ),( jnp.sin( 1/2 * jnp.real( phi ) ) )**( 2 ),] ),jnp.array( [0,0,0,0,1/2 * ( jnp.cos( 1/2 * phi ) + jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),complex( 0,-1/2 ) * ( jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),0,0,complex( 0,1/2 ) * ( jnp.cos( 1/2 * phi ) + -1 * jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),1/2 * ( jnp.sin( 1/2 * phi ) + jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),0,0,0,0,0,0,] ),jnp.array( [0,0,0,0,complex( 0,-1/2 ) * ( jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),1/2 * ( jnp.cos( 1/2 * phi ) + jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),0,0,1/2 * ( jnp.sin( 1/2 * phi ) + jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),complex( 0,1/2 ) * ( jnp.cos( 1/2 * phi ) + -1 * jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),0,0,0,0,0,0,] ),jnp.array( [0,0,0,0,0,0,1/2 * ( jnp.cos( 1/2 * phi ) + jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),1/2 * ( -1 * jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),0,0,complex( 0,1/2 ) * ( jnp.cos( 1/2 * phi ) + -1 * jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),complex( 0,-1/2 ) * ( jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),0,0,0,0,] ),jnp.array( [0,0,0,0,0,0,1/2 * ( jnp.sin( 1/2 * phi ) + jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),1/2 * ( jnp.cos( 1/2 * phi ) + jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),0,0,complex( 0,1/2 ) * ( jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),complex( 0,1/2 ) * ( jnp.cos( 1/2 * phi ) + -1 * jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),0,0,0,0,] ),jnp.array( [0,0,0,0,complex( 0,-1/2 ) * ( jnp.cos( 1/2 * phi ) + -1 * jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),1/2 * ( -1 * jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),0,0,1/2 * ( jnp.cos( 1/2 * phi ) + jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),complex( 0,-1/2 ) * ( jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),0,0,0,0,0,0,] ),jnp.array( [0,0,0,0,1/2 * ( -1 * jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),complex( 0,-1/2 ) * ( jnp.cos( 1/2 * phi ) + -1 * jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),0,0,complex( 0,-1/2 ) * ( jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),1/2 * ( jnp.cos( 1/2 * phi ) + jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),0,0,0,0,0,0,] ),jnp.array( [0,0,0,0,0,0,complex( 0,-1/2 ) * ( jnp.cos( 1/2 * phi ) + -1 * jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),complex( 0,1/2 ) * ( jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),0,0,1/2 * ( jnp.cos( 1/2 * phi ) + jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),1/2 * ( -1 * jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),0,0,0,0,] ),jnp.array( [0,0,0,0,0,0,complex( 0,-1/2 ) * ( jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),complex( 0,-1/2 ) * ( jnp.cos( 1/2 * phi ) + -1 * jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),0,0,1/2 * ( jnp.sin( 1/2 * phi ) + jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),1/2 * ( jnp.cos( 1/2 * phi ) + jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),0,0,0,0,] ),jnp.array( [-1 * ( jnp.sinh( 1/2 * jnp.imag( phi ) ) )**( 2 ),-1/2 * jnp.sinh( jnp.imag( phi ) ),0,0,0,0,0,0,0,0,0,0,( jnp.cosh( 1/2 * jnp.imag( phi ) ) )**( 2 ),1/2 * jnp.sinh( jnp.imag( phi ) ),0,0,] ),jnp.array( [-1/2 * jnp.sinh( jnp.imag( phi ) ),-1 * ( jnp.sinh( 1/2 * jnp.imag( phi ) ) )**( 2 ),0,0,0,0,0,0,0,0,0,0,1/2 * jnp.sinh( jnp.imag( phi ) ),( jnp.cosh( 1/2 * jnp.imag( phi ) ) )**( 2 ),0,0,] ),jnp.array( [0,0,( jnp.sin( 1/2 * jnp.real( phi ) ) )**( 2 ),1/2 * jnp.sin( jnp.real( phi ) ),0,0,0,0,0,0,0,0,0,0,( jnp.cos( 1/2 * jnp.real( phi ) ) )**( 2 ),-1/2 * jnp.sin( jnp.real( phi ) ),] ),jnp.array( [0,0,-1/2 * jnp.sin( jnp.real( phi ) ),( jnp.sin( 1/2 * jnp.real( phi ) ) )**( 2 ),0,0,0,0,0,0,0,0,0,0,1/2 * jnp.sin( jnp.real( phi ) ),( jnp.cos( 1/2 * jnp.real( phi ) ) )**( 2 ),] ),] )
        
        elif self.name == "CRY":
            if self.params is None:
                raise ValueError("Parameter 'params' must be provided for CRY gate.")
            phi =  self.params[0]
            return jnp.array( [jnp.array( [( jnp.cosh( 1/2 * jnp.imag( phi ) ) )**( 2 ),0,1/2 * jnp.sinh( jnp.imag( phi ) ),0,0,0,0,0,0,0,0,0,-1 * ( jnp.sinh( 1/2 * jnp.imag( phi ) ) )**( 2 ),0,-1/2 * jnp.sinh( jnp.imag( phi ) ),0,] ),jnp.array( [0,( jnp.cos( 1/2 * jnp.real( phi ) ) )**( 2 ),0,1/2 * jnp.sin( jnp.real( phi ) ),0,0,0,0,0,0,0,0,0,( jnp.sin( 1/2 * jnp.real( phi ) ) )**( 2 ),0,-1/2 * jnp.sin( jnp.real( phi ) ),] ),jnp.array( [1/2 * jnp.sinh( jnp.imag( phi ) ),0,( jnp.cosh( 1/2 * jnp.imag( phi ) ) )**( 2 ),0,0,0,0,0,0,0,0,0,-1/2 * jnp.sinh( jnp.imag( phi ) ),0,-1 * ( jnp.sinh( 1/2 * jnp.imag( phi ) ) )**( 2 ),0,] ),jnp.array( [0,-1/2 * jnp.sin( jnp.real( phi ) ),0,( jnp.cos( 1/2 * jnp.real( phi ) ) )**( 2 ),0,0,0,0,0,0,0,0,0,1/2 * jnp.sin( jnp.real( phi ) ),0,( jnp.sin( 1/2 * jnp.real( phi ) ) )**( 2 ),] ),jnp.array( [0,0,0,0,1/2 * ( jnp.cos( 1/2 * phi ) + jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),0,complex( 0,-1/2 ) * ( jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),0,complex( 0,1/2 ) * ( jnp.cos( 1/2 * phi ) + -1 * jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),0,1/2 * ( jnp.sin( 1/2 * phi ) + jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),0,0,0,0,0,] ),jnp.array( [0,0,0,0,0,1/2 * ( jnp.cos( 1/2 * phi ) + jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),0,1/2 * ( jnp.sin( 1/2 * phi ) + jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),0,complex( 0,1/2 ) * ( jnp.cos( 1/2 * phi ) + -1 * jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),0,complex( 0,1/2 ) * ( jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),0,0,0,0,] ),jnp.array( [0,0,0,0,complex( 0,-1/2 ) * ( jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),0,1/2 * ( jnp.cos( 1/2 * phi ) + jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),0,1/2 * ( jnp.sin( 1/2 * phi ) + jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),0,complex( 0,1/2 ) * ( jnp.cos( 1/2 * phi ) + -1 * jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),0,0,0,0,0,] ),jnp.array( [0,0,0,0,0,1/2 * ( -1 * jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),0,1/2 * ( jnp.cos( 1/2 * phi ) + jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),0,complex( 0,-1/2 ) * ( jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),0,complex( 0,1/2 ) * ( jnp.cos( 1/2 * phi ) + -1 * jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),0,0,0,0,] ),jnp.array( [0,0,0,0,complex( 0,-1/2 ) * ( jnp.cos( 1/2 * phi ) + -1 * jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),0,1/2 * ( -1 * jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),0,1/2 * ( jnp.cos( 1/2 * phi ) + jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),0,complex( 0,-1/2 ) * ( jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),0,0,0,0,0,] ),jnp.array( [0,0,0,0,0,complex( 0,-1/2 ) * ( jnp.cos( 1/2 * phi ) + -1 * jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),0,complex( 0,-1/2 ) * ( jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),0,1/2 * ( jnp.cos( 1/2 * phi ) + jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),0,1/2 * ( jnp.sin( 1/2 * phi ) + jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),0,0,0,0,] ),jnp.array( [0,0,0,0,1/2 * ( -1 * jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),0,complex( 0,-1/2 ) * ( jnp.cos( 1/2 * phi ) + -1 * jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),0,complex( 0,-1/2 ) * ( jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),0,1/2 * ( jnp.cos( 1/2 * phi ) + jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),0,0,0,0,0,] ),jnp.array( [0,0,0,0,0,complex( 0,1/2 ) * ( jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),0,complex( 0,-1/2 ) * ( jnp.cos( 1/2 * phi ) + -1 * jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),0,1/2 * ( -1 * jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),0,1/2 * ( jnp.cos( 1/2 * phi ) + jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),0,0,0,0,] ),jnp.array( [-1 * ( jnp.sinh( 1/2 * jnp.imag( phi ) ) )**( 2 ),0,-1/2 * jnp.sinh( jnp.imag( phi ) ),0,0,0,0,0,0,0,0,0,( jnp.cosh( 1/2 * jnp.imag( phi ) ) )**( 2 ),0,1/2 * jnp.sinh( jnp.imag( phi ) ),0,] ),jnp.array( [0,( jnp.sin( 1/2 * jnp.real( phi ) ) )**( 2 ),0,-1/2 * jnp.sin( jnp.real( phi ) ),0,0,0,0,0,0,0,0,0,( jnp.cos( 1/2 * jnp.real( phi ) ) )**( 2 ),0,1/2 * jnp.sin( jnp.real( phi ) ),] ),jnp.array( [-1/2 * jnp.sinh( jnp.imag( phi ) ),0,-1 * ( jnp.sinh( 1/2 * jnp.imag( phi ) ) )**( 2 ),0,0,0,0,0,0,0,0,0,1/2 * jnp.sinh( jnp.imag( phi ) ),0,( jnp.cosh( 1/2 * jnp.imag( phi ) ) )**( 2 ),0,] ),jnp.array( [0,1/2 * jnp.sin( jnp.real( phi ) ),0,( jnp.sin( 1/2 * jnp.real( phi ) ) )**( 2 ),0,0,0,0,0,0,0,0,0,-1/2 * jnp.sin( jnp.real( phi ) ),0,( jnp.cos( 1/2 * jnp.real( phi ) ) )**( 2 ),] ),] )
        
        elif self.name == "CRZ":
            if self.params is None:
                raise ValueError("Parameter 'params' must be provided for CRZ gate.")
            phi =  self.params[0]
            return jnp.array( [jnp.array( [1/4 * ( jnp.e )**( complex( 0,-1 ) * jnp.real( phi ) ) * ( ( ( jnp.e )**( complex( 0,1/2 ) * phi ) + ( jnp.e )**( complex( 0,1/2 ) * jnp.conjugate( phi ) ) ) )**( 2 ),0,0,1/4 * ( jnp.e )**( complex( 0,-1 ) * jnp.real( phi ) ) * ( -1 * ( jnp.e )**( complex( 0,1 ) * phi ) + ( jnp.e )**( complex( 0,1 ) * jnp.conjugate( phi ) ) ),0,0,0,0,0,0,0,0,-1/4 * ( jnp.e )**( complex( 0,-1 ) * jnp.real( phi ) ) * ( ( ( jnp.e )**( complex( 0,1/2 ) * phi ) + -1 * ( jnp.e )**( complex( 0,1/2 ) * jnp.conjugate( phi ) ) ) )**( 2 ),0,0,1/4 * ( jnp.e )**( complex( 0,-1 ) * jnp.real( phi ) ) * ( ( jnp.e )**( complex( 0,1 ) * phi ) + -1 * ( jnp.e )**( complex( 0,1 ) * jnp.conjugate( phi ) ) ),] ),jnp.array( [0,( jnp.cos( 1/2 * jnp.real( phi ) ) )**( 2 ),-1/2 * jnp.sin( jnp.real( phi ) ),0,0,0,0,0,0,0,0,0,0,( jnp.sin( 1/2 * jnp.real( phi ) ) )**( 2 ),1/2 * jnp.sin( jnp.real( phi ) ),0,] ),jnp.array( [0,1/2 * jnp.sin( jnp.real( phi ) ),( jnp.cos( 1/2 * jnp.real( phi ) ) )**( 2 ),0,0,0,0,0,0,0,0,0,0,-1/2 * jnp.sin( jnp.real( phi ) ),( jnp.sin( 1/2 * jnp.real( phi ) ) )**( 2 ),0,] ),jnp.array( [1/4 * ( jnp.e )**( complex( 0,-1 ) * jnp.real( phi ) ) * ( -1 * ( jnp.e )**( complex( 0,1 ) * phi ) + ( jnp.e )**( complex( 0,1 ) * jnp.conjugate( phi ) ) ),0,0,1/4 * ( jnp.e )**( complex( 0,-1 ) * jnp.real( phi ) ) * ( ( ( jnp.e )**( complex( 0,1/2 ) * phi ) + ( jnp.e )**( complex( 0,1/2 ) * jnp.conjugate( phi ) ) ) )**( 2 ),0,0,0,0,0,0,0,0,1/4 * ( jnp.e )**( complex( 0,-1 ) * jnp.real( phi ) ) * ( ( jnp.e )**( complex( 0,1 ) * phi ) + -1 * ( jnp.e )**( complex( 0,1 ) * jnp.conjugate( phi ) ) ),0,0,-1/4 * ( jnp.e )**( complex( 0,-1 ) * jnp.real( phi ) ) * ( ( ( jnp.e )**( complex( 0,1/2 ) * phi ) + -1 * ( jnp.e )**( complex( 0,1/2 ) * jnp.conjugate( phi ) ) ) )**( 2 ),] ),jnp.array( [0,0,0,0,1/2 * ( jnp.cos( 1/2 * phi ) + jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),0,0,complex( 0,-1/2 ) * ( jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),complex( 0,1/2 ) * ( jnp.cos( 1/2 * phi ) + -1 * jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),0,0,1/2 * ( jnp.sin( 1/2 * phi ) + jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),0,0,0,0,] ),jnp.array( [0,0,0,0,0,1/2 * ( jnp.cos( 1/2 * phi ) + jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),1/2 * ( -1 * jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),0,0,complex( 0,1/2 ) * ( jnp.cos( 1/2 * phi ) + -1 * jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),complex( 0,-1/2 ) * ( jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),0,0,0,0,0,] ),jnp.array( [0,0,0,0,0,1/2 * ( jnp.sin( 1/2 * phi ) + jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),1/2 * ( jnp.cos( 1/2 * phi ) + jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),0,0,1/4 * ( jnp.e )**( complex( 0,-1 ) * jnp.real( phi ) ) * ( ( jnp.e )**( complex( 0,1/2 ) * phi ) + -1 * ( jnp.e )**( complex( 0,1/2 ) * jnp.conjugate( phi ) ) ) * ( 1 + ( jnp.e )**( complex( 0,1 ) * jnp.real( phi ) ) ),complex( 0,1/2 ) * ( jnp.cos( 1/2 * phi ) + -1 * jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),0,0,0,0,0,] ),jnp.array( [0,0,0,0,complex( 0,-1/2 ) * ( jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),0,0,1/2 * ( jnp.cos( 1/2 * phi ) + jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),1/2 * ( jnp.sin( 1/2 * phi ) + jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),0,0,complex( 0,1/2 ) * ( jnp.cos( 1/2 * phi ) + -1 * jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),0,0,0,0,] ),jnp.array( [0,0,0,0,complex( 0,-1/2 ) * ( jnp.cos( 1/2 * phi ) + -1 * jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),0,0,1/2 * ( -1 * jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),1/2 * ( jnp.cos( 1/2 * phi ) + jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),0,0,complex( 0,-1/2 ) * ( jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),0,0,0,0,] ),jnp.array( [0,0,0,0,0,complex( 0,-1/2 ) * ( jnp.cos( 1/2 * phi ) + -1 * jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),1/4 * ( jnp.e )**( complex( 0,-1 ) * jnp.real( phi ) ) * ( ( jnp.e )**( complex( 0,1/2 ) * phi ) + -1 * ( jnp.e )**( complex( 0,1/2 ) * jnp.conjugate( phi ) ) ) * ( 1 + ( jnp.e )**( complex( 0,1 ) * jnp.real( phi ) ) ),0,0,1/2 * ( jnp.cos( 1/2 * phi ) + jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),1/2 * ( -1 * jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),0,0,0,0,0,] ),jnp.array( [0,0,0,0,0,complex( 0,-1/2 ) * ( jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),complex( 0,-1/2 ) * ( jnp.cos( 1/2 * phi ) + -1 * jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),0,0,1/2 * ( jnp.sin( 1/2 * phi ) + jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),1/2 * ( jnp.cos( 1/2 * phi ) + jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),0,0,0,0,0,] ),jnp.array( [0,0,0,0,1/2 * ( -1 * jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),0,0,complex( 0,-1/2 ) * ( jnp.cos( 1/2 * phi ) + -1 * jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),complex( 0,-1/2 ) * ( jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),0,0,1/2 * ( jnp.cos( 1/2 * phi ) + jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),0,0,0,0,] ),jnp.array( [-1/4 * ( jnp.e )**( complex( 0,-1 ) * jnp.real( phi ) ) * ( ( ( jnp.e )**( complex( 0,1/2 ) * phi ) + -1 * ( jnp.e )**( complex( 0,1/2 ) * jnp.conjugate( phi ) ) ) )**( 2 ),0,0,1/4 * ( jnp.e )**( complex( 0,-1 ) * jnp.real( phi ) ) * ( ( jnp.e )**( complex( 0,1 ) * phi ) + -1 * ( jnp.e )**( complex( 0,1 ) * jnp.conjugate( phi ) ) ),0,0,0,0,0,0,0,0,1/4 * ( jnp.e )**( complex( 0,-1 ) * jnp.real( phi ) ) * ( ( ( jnp.e )**( complex( 0,1/2 ) * phi ) + ( jnp.e )**( complex( 0,1/2 ) * jnp.conjugate( phi ) ) ) )**( 2 ),0,0,1/4 * ( jnp.e )**( complex( 0,-1 ) * jnp.real( phi ) ) * ( -1 * ( jnp.e )**( complex( 0,1 ) * phi ) + ( jnp.e )**( complex( 0,1 ) * jnp.conjugate( phi ) ) ),] ),jnp.array( [0,( jnp.sin( 1/2 * jnp.real( phi ) ) )**( 2 ),1/2 * jnp.sin( jnp.real( phi ) ),0,0,0,0,0,0,0,0,0,0,( jnp.cos( 1/2 * jnp.real( phi ) ) )**( 2 ),-1/2 * jnp.sin( jnp.real( phi ) ),0,] ),jnp.array( [0,-1/2 * jnp.sin( jnp.real( phi ) ),( jnp.sin( 1/2 * jnp.real( phi ) ) )**( 2 ),0,0,0,0,0,0,0,0,0,0,1/2 * jnp.sin( jnp.real( phi ) ),( jnp.cos( 1/2 * jnp.real( phi ) ) )**( 2 ),0,] ),jnp.array( [1/4 * ( jnp.e )**( complex( 0,-1 ) * jnp.real( phi ) ) * ( ( jnp.e )**( complex( 0,1 ) * phi ) + -1 * ( jnp.e )**( complex( 0,1 ) * jnp.conjugate( phi ) ) ),0,0,-1/4 * ( jnp.e )**( complex( 0,-1 ) * jnp.real( phi ) ) * ( ( ( jnp.e )**( complex( 0,1/2 ) * phi ) + -1 * ( jnp.e )**( complex( 0,1/2 ) * jnp.conjugate( phi ) ) ) )**( 2 ),0,0,0,0,0,0,0,0,1/4 * ( jnp.e )**( complex( 0,-1 ) * jnp.real( phi ) ) * ( -1 * ( jnp.e )**( complex( 0,1 ) * phi ) + ( jnp.e )**( complex( 0,1 ) * jnp.conjugate( phi ) ) ),0,0,1/4 * ( jnp.e )**( complex( 0,-1 ) * jnp.real( phi ) ) * ( ( ( jnp.e )**( complex( 0,1/2 ) * phi ) + ( jnp.e )**( complex( 0,1/2 ) * jnp.conjugate( phi ) ) ) )**( 2 ),] ),] )
        elif self.name == "CRY_S":
            #CRY followed by S 
            if self.params is None:
                raise ValueError("Parameter 'params' must be provided for CRY_S gate.")
            phi =  self.params[0]
            return jnp.array( [jnp.array( [( jnp.cosh( 1/2 * jnp.imag( phi ) ) )**( 2 ),0,1/2 * jnp.sinh( jnp.imag( phi ) ),0,0,0,0,0,0,0,0,0,-1 * ( jnp.sinh( 1/2 * jnp.imag( phi ) ) )**( 2 ),0,-1/2 * jnp.sinh( jnp.imag( phi ) ),0,] ),jnp.array( [0,( jnp.cos( 1/2 * jnp.real( phi ) ) )**( 2 ),0,1/2 * jnp.sin( jnp.real( phi ) ),0,0,0,0,0,0,0,0,0,( jnp.sin( 1/2 * jnp.real( phi ) ) )**( 2 ),0,-1/2 * jnp.sin( jnp.real( phi ) ),] ),jnp.array( [1/2 * jnp.sinh( jnp.imag( phi ) ),0,( jnp.cosh( 1/2 * jnp.imag( phi ) ) )**( 2 ),0,0,0,0,0,0,0,0,0,-1/2 * jnp.sinh( jnp.imag( phi ) ),0,-1 * ( jnp.sinh( 1/2 * jnp.imag( phi ) ) )**( 2 ),0,] ),jnp.array( [0,-1/2 * jnp.sin( jnp.real( phi ) ),0,( jnp.cos( 1/2 * jnp.real( phi ) ) )**( 2 ),0,0,0,0,0,0,0,0,0,1/2 * jnp.sin( jnp.real( phi ) ),0,( jnp.sin( 1/2 * jnp.real( phi ) ) )**( 2 ),] ),jnp.array( [0,0,0,0,complex( 0,-1/2 ) * ( jnp.cos( 1/2 * phi ) + -1 * jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),0,1/2 * ( -1 * jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),0,1/2 * ( jnp.cos( 1/2 * phi ) + jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),0,complex( 0,-1/2 ) * ( jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),0,0,0,0,0,] ),jnp.array( [0,0,0,0,0,complex( 0,-1/2 ) * ( jnp.cos( 1/2 * phi ) + -1 * jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),0,complex( 0,-1/2 ) * ( jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),0,1/2 * ( jnp.cos( 1/2 * phi ) + jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),0,1/2 * ( jnp.sin( 1/2 * phi ) + jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),0,0,0,0,] ),jnp.array( [0,0,0,0,1/2 * ( -1 * jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),0,complex( 0,-1/2 ) * ( jnp.cos( 1/2 * phi ) + -1 * jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),0,complex( 0,-1/2 ) * ( jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),0,1/2 * ( jnp.cos( 1/2 * phi ) + jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),0,0,0,0,0,] ),jnp.array( [0,0,0,0,0,complex( 0,1/2 ) * ( jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),0,complex( 0,-1/2 ) * ( jnp.cos( 1/2 * phi ) + -1 * jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),0,1/2 * ( -1 * jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),0,1/2 * ( jnp.cos( 1/2 * phi ) + jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),0,0,0,0,] ),jnp.array( [0,0,0,0,1/2 * ( -1 * jnp.cos( 1/2 * phi ) + -1 * jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),0,complex( 0,1/2 ) * ( jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),0,complex( 0,-1/2 ) * ( jnp.cos( 1/2 * phi ) + -1 * jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),0,1/2 * ( -1 * jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),0,0,0,0,0,] ),jnp.array( [0,0,0,0,0,1/2 * ( -1 * jnp.cos( 1/2 * phi ) + -1 * jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),0,1/2 * ( -1 * jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),0,complex( 0,-1/2 ) * ( jnp.cos( 1/2 * phi ) + -1 * jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),0,complex( 0,-1/2 ) * ( jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),0,0,0,0,] ),jnp.array( [0,0,0,0,complex( 0,1/2 ) * ( jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),0,1/2 * ( -1 * jnp.cos( 1/2 * phi ) + -1 * jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),0,1/2 * ( -1 * jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),0,complex( 0,-1/2 ) * ( jnp.cos( 1/2 * phi ) + -1 * jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),0,0,0,0,0,] ),jnp.array( [0,0,0,0,0,1/2 * ( jnp.sin( 1/2 * phi ) + jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),0,1/2 * ( -1 * jnp.cos( 1/2 * phi ) + -1 * jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),0,complex( 0,1/2 ) * ( jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),0,complex( 0,-1/2 ) * ( jnp.cos( 1/2 * phi ) + -1 * jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),0,0,0,0,] ),jnp.array( [-1 * ( jnp.sinh( 1/2 * jnp.imag( phi ) ) )**( 2 ),0,-1/2 * jnp.sinh( jnp.imag( phi ) ),0,0,0,0,0,0,0,0,0,( jnp.cosh( 1/2 * jnp.imag( phi ) ) )**( 2 ),0,1/2 * jnp.sinh( jnp.imag( phi ) ),0,] ),jnp.array( [0,( jnp.sin( 1/2 * jnp.real( phi ) ) )**( 2 ),0,-1/2 * jnp.sin( jnp.real( phi ) ),0,0,0,0,0,0,0,0,0,( jnp.cos( 1/2 * jnp.real( phi ) ) )**( 2 ),0,1/2 * jnp.sin( jnp.real( phi ) ),] ),jnp.array( [-1/2 * jnp.sinh( jnp.imag( phi ) ),0,-1 * ( jnp.sinh( 1/2 * jnp.imag( phi ) ) )**( 2 ),0,0,0,0,0,0,0,0,0,1/2 * jnp.sinh( jnp.imag( phi ) ),0,( jnp.cosh( 1/2 * jnp.imag( phi ) ) )**( 2 ),0,] ),jnp.array( [0,1/2 * jnp.sin( jnp.real( phi ) ),0,( jnp.sin( 1/2 * jnp.real( phi ) ) )**( 2 ),0,0,0,0,0,0,0,0,0,-1/2 * jnp.sin( jnp.real( phi ) ),0,( jnp.cos( 1/2 * jnp.real( phi ) ) )**( 2 ),] ),] )
        if self.name == "CRX_S":
            #CRY followed by S
            if self.params is None:
                raise ValueError("Parameter 'params' must be provided for CRX_S gate.")
            phi =  self.params[0]
            return jnp.array( [jnp.array( [( jnp.cosh( 1/2 * jnp.imag( phi ) ) )**( 2 ),1/2 * jnp.sinh( jnp.imag( phi ) ),0,0,0,0,0,0,0,0,0,0,-1 * ( jnp.sinh( 1/2 * jnp.imag( phi ) ) )**( 2 ),-1/2 * jnp.sinh( jnp.imag( phi ) ),0,0,] ),jnp.array( [1/2 * jnp.sinh( jnp.imag( phi ) ),( jnp.cosh( 1/2 * jnp.imag( phi ) ) )**( 2 ),0,0,0,0,0,0,0,0,0,0,-1/2 * jnp.sinh( jnp.imag( phi ) ),-1 * ( jnp.sinh( 1/2 * jnp.imag( phi ) ) )**( 2 ),0,0,] ),jnp.array( [0,0,( jnp.cos( 1/2 * jnp.real( phi ) ) )**( 2 ),-1/2 * jnp.sin( jnp.real( phi ) ),0,0,0,0,0,0,0,0,0,0,( jnp.sin( 1/2 * jnp.real( phi ) ) )**( 2 ),1/2 * jnp.sin( jnp.real( phi ) ),] ),jnp.array( [0,0,1/2 * jnp.sin( jnp.real( phi ) ),( jnp.cos( 1/2 * jnp.real( phi ) ) )**( 2 ),0,0,0,0,0,0,0,0,0,0,-1/2 * jnp.sin( jnp.real( phi ) ),( jnp.sin( 1/2 * jnp.real( phi ) ) )**( 2 ),] ),jnp.array( [0,0,0,0,complex( 0,-1/2 ) * ( jnp.cos( 1/2 * phi ) + -1 * jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),1/2 * ( -1 * jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),0,0,1/2 * ( jnp.cos( 1/2 * phi ) + jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),complex( 0,-1/2 ) * ( jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),0,0,0,0,0,0,] ),jnp.array( [0,0,0,0,1/2 * ( -1 * jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),complex( 0,-1/2 ) * ( jnp.cos( 1/2 * phi ) + -1 * jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),0,0,complex( 0,-1/2 ) * ( jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),1/2 * ( jnp.cos( 1/2 * phi ) + jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),0,0,0,0,0,0,] ),jnp.array( [0,0,0,0,0,0,complex( 0,-1/2 ) * ( jnp.cos( 1/2 * phi ) + -1 * jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),complex( 0,1/2 ) * ( jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),0,0,1/2 * ( jnp.cos( 1/2 * phi ) + jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),1/2 * ( -1 * jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),0,0,0,0,] ),jnp.array( [0,0,0,0,0,0,complex( 0,-1/2 ) * ( jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),complex( 0,-1/2 ) * ( jnp.cos( 1/2 * phi ) + -1 * jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),0,0,1/2 * ( jnp.sin( 1/2 * phi ) + jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),1/2 * ( jnp.cos( 1/2 * phi ) + jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),0,0,0,0,] ),jnp.array( [0,0,0,0,1/2 * ( -1 * jnp.cos( 1/2 * phi ) + -1 * jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),complex( 0,1/2 ) * ( jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),0,0,complex( 0,-1/2 ) * ( jnp.cos( 1/2 * phi ) + -1 * jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),1/2 * ( -1 * jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),0,0,0,0,0,0,] ),jnp.array( [0,0,0,0,complex( 0,1/2 ) * ( jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),1/2 * ( -1 * jnp.cos( 1/2 * phi ) + -1 * jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),0,0,1/2 * ( -1 * jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),complex( 0,-1/2 ) * ( jnp.cos( 1/2 * phi ) + -1 * jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),0,0,0,0,0,0,] ),jnp.array( [0,0,0,0,0,0,1/2 * ( -1 * jnp.cos( 1/2 * phi ) + -1 * jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),1/2 * ( jnp.sin( 1/2 * phi ) + jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),0,0,complex( 0,-1/2 ) * ( jnp.cos( 1/2 * phi ) + -1 * jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),complex( 0,1/2 ) * ( jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),0,0,0,0,] ),jnp.array( [0,0,0,0,0,0,1/2 * ( -1 * jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),1/2 * ( -1 * jnp.cos( 1/2 * phi ) + -1 * jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),0,0,complex( 0,-1/2 ) * ( jnp.sin( 1/2 * phi ) + -1 * jnp.sin( 1/2 * jnp.conjugate( phi ) ) ),complex( 0,-1/2 ) * ( jnp.cos( 1/2 * phi ) + -1 * jnp.cos( 1/2 * jnp.conjugate( phi ) ) ),0,0,0,0,] ),jnp.array( [-1 * ( jnp.sinh( 1/2 * jnp.imag( phi ) ) )**( 2 ),-1/2 * jnp.sinh( jnp.imag( phi ) ),0,0,0,0,0,0,0,0,0,0,( jnp.cosh( 1/2 * jnp.imag( phi ) ) )**( 2 ),1/2 * jnp.sinh( jnp.imag( phi ) ),0,0,] ),jnp.array( [-1/2 * jnp.sinh( jnp.imag( phi ) ),-1 * ( jnp.sinh( 1/2 * jnp.imag( phi ) ) )**( 2 ),0,0,0,0,0,0,0,0,0,0,1/2 * jnp.sinh( jnp.imag( phi ) ),( jnp.cosh( 1/2 * jnp.imag( phi ) ) )**( 2 ),0,0,] ),jnp.array( [0,0,( jnp.sin( 1/2 * jnp.real( phi ) ) )**( 2 ),1/2 * jnp.sin( jnp.real( phi ) ),0,0,0,0,0,0,0,0,0,0,( jnp.cos( 1/2 * jnp.real( phi ) ) )**( 2 ),-1/2 * jnp.sin( jnp.real( phi ) ),] ),jnp.array( [0,0,-1/2 * jnp.sin( jnp.real( phi ) ),( jnp.sin( 1/2 * jnp.real( phi ) ) )**( 2 ),0,0,0,0,0,0,0,0,0,0,1/2 * jnp.sin( jnp.real( phi ) ),( jnp.cos( 1/2 * jnp.real( phi ) ) )**( 2 ),] ),] )
        
        ###########################################################
        ###################     Channels      #####################
        ###########################################################

        elif self.name == "AmplitudeDamping":
            
            if self.params is None:
                raise ValueError("Parameter 'params' must be provided for Ampdamp channel.")
            g =  self.params[0]
            return jnp.array( [jnp.array( [1,0,0,0,] ),jnp.array( [0,( ( 1 + -1 * g ) )**( 1/2 ),0,0,] ),jnp.array( [0,0,( ( 1 + -1 * g ) )**( 1/2 ),0,] ),jnp.array( [g,0,0,( 1 + -1 * g ),] ),] )
        
        elif self.name == "Depolarizing":
            if self.params is None:
                raise ValueError("Parameter 'params' must be provided for Depolarizing channel.")
            p =  self.params[0]
            return jnp.array( [jnp.array( [( jnp.abs( ( 1 + -1 * p ) ) + jnp.abs( p ) ),0,0,0,] ),jnp.array( [0,( jnp.abs( ( 1 + -1 * p ) ) + -1/3 * jnp.abs( p ) ),0,0,] ),jnp.array( [0,0,( jnp.abs( ( 1 + -1 * p ) ) + -1/3 * jnp.abs( p ) ),0,] ),jnp.array( [0,0,0,( jnp.abs( ( 1 + -1 * p ) ) + -1/3 * jnp.abs( p ) ),] ),] )
        else:
            raise ValueError("Gate " + self.name + " not supported." )
    
    def _get_rmatrix(self):
        """Returns the matrix representation as affine transformation v' = M v + c.

        Returns:
            tuple: The matrix M and vector c.
        """
        M = self._get_matrix()
        c = M[1:,0]
        Mred = M[1:,1:]
        return Mred,c/jnp.sqrt(2**(jnp.log2(jnp.shape(c)[0]+1)/2)) 
    def _is_nonunital(self):
        """Checks if the operation is unital.

        Returns:
            bool: True if the operation is nonunital, False otherwise.
        """
        M,c = self._get_rmatrix()
        tol = 1e-9
        if jnp.allclose(c, 0, atol=tol):
            return False
        else: 
            return True
        
    def __repr__(self):
        return f"Gate {self.name}" 
    
    def circuit_gate(self):
        """Takes the Gate and makes it to a N_qubits operation put at location Gate_loc.

        Returns:
            ndarray: The circuit gate matrix.
        """
        GM = self._get_matrix()
        if self.Gate_loc is None:
            raise ValueError("To create a circuit gate need to specify the location in the initialization of the gate.")
        else:
            Gate_loc = self.Gate_loc

        if self.N_qubits is None:
            raise ValueError("To create a circuit gate need to specify the number of qubits in the initialization of the gate.")
        else:
            N_qubits = self.N_qubits
        ############# Single qubit gates 
        if GM.shape == (4,4):
            if isinstance(Gate_loc, int):
                l = Gate_loc
                pattern = False
            # Check if Gate_loc is a list with a single entry
            elif (isinstance(Gate_loc, list) or isinstance(Gate_loc,jnp.ndarray)) and len(Gate_loc) == 1:
                l = Gate_loc[0]
                pattern = False
            else:
                pattern = True
                #raise ValueError("Need a single location for single qubit gate. You gave shape"+str(jnp.shape(Gate_loc))+ "for the gate "+self.name)
            if pattern: 
                #initialize check if the gate is applied to the first qubit
                if 0 in Gate_loc:
                    CM = GM
                    i = 1
                else: 
                    CM = jnp.eye(4**1)
                    i = 0
                #go through pattern and add the correct gates to the matrix
                for j in range(1,self.N_qubits): 
                    if j in Gate_loc[i:]:
                        CM = jnp.kron(CM,GM)
                    else: 
                        CM = jnp.kron(CM,jnp.eye(4))
                return CM 
            else: 
                if  N_qubits <= l:
                    raise ValueError("Index cannot be larger than the number of qubits Index = ", l, "N_qubits = ", N_qubits)
                if l == 0:
                    #on the first qubit
                    return jnp.kron(GM,jnp.eye(4**(N_qubits-1)))
                elif l == N_qubits-1:
                    #on the last qubit
                    return jnp.kron(jnp.eye(4**(N_qubits-1)),GM)
                else:
                    # in the middle
                    return jnp.kron(jnp.kron(jnp.eye(4**(l)),GM),jnp.eye(4**(N_qubits-l-1)))

        
        ############ two qubit gates 
        if GM.shape == (16,16):
            if N_qubits < 2:
                raise ValueError("Number of qubits must be at least 2")
            
            elif (isinstance(Gate_loc, list) or isinstance(Gate_loc,jnp.ndarray))  and jnp.shape(Gate_loc)[0] == 2:
                l,m = Gate_loc
                if l < 0 or l >= N_qubits or m < 0 or m >= N_qubits:
                    print("expanding")
                    raise ValueError("Invalid qubit indices")
                return expand_gate_Bloch(GM, [l,m], N_qubits)
            else: 
                #WARNING: So far the pattern has to be of neighboring gates of ascending order. To make it work in generality would need some permutation. 
                len_pattern = len(Gate_loc)
                print("broadcasting")
                # Generate the ops that start it 
                if self.pattern == "double" and len_pattern%2 == 0:
                    neighbour_op = kron_multi(GM,int(len_pattern/2))
                elif self.pattern == "double" and len_pattern%2 == 1:
                    neighbour_op = jnp.kron(kron_multi(GM,int(len_pattern/2-1)),jnp.eye(4))
                elif self.pattern == "double_odd" and len_pattern%2 == 0: 
                    neighbour_op = jnp.kron(jnp.kron(jnp.eye(4),kron_multi(GM,int(len_pattern/2-1))),jnp.eye(4))
                    print(np.shape(neighbour_op),np.shape(expand_gate_Bloch(GM, [Gate_loc[-1],Gate_loc[0]], N_qubits)))
                    neighbour_op = neighbour_op  @ expand_gate_Bloch(GM, [Gate_loc[-1],Gate_loc[0]], len_pattern)
                    neworder = (np.arange(len_pattern)+1)%len_pattern
                elif self.pattern == "double_odd" and len_pattern%2 == 1: 
                    neighbour_op = jnp.kron(jnp.eye(4),kron_multi(GM,int(np.ceil(len_pattern/2-1))))
                else: 
                    raise ValueError("Unknown pattern: "+ pattern)
                if len_pattern != N_qubits:
                    #pad with identies
                    l = Gate_loc[0]
                    if Gate_loc[0] == 0:
                        #on the first qubit
                        return jnp.kron(neighbour_op,jnp.eye(4**(N_qubits-Gate_loc[-1]-1)))
                    elif Gate_loc[-1] == N_qubits-1:
                        #on the last qubit
                        return jnp.kron(jnp.eye(4**(Gate_loc[0])),neighbour_op)
                    else:
                        #middle
                        return  jnp.kron(jnp.eye(4**Gate_loc[0]), jnp.kron(neighbour_op,jnp.eye(4**(self.N_qubits-Gate_loc[-1]-1))))
                    
                return neighbour_op

 
    def rcircuit_gate(self):
        """Returns the reduced circuit gate matrix.

        Returns:
            tuple: The reduced circuit gate matrix and vector.
        """
        cg = self.circuit_gate()
        Omega = cg[1:,1:]
        c = cg[1:,0]
        return Omega,c/jnp.sqrt(2**(jnp.log2(jnp.shape(c)[0]+1)/2))



# Helper function to generate Pauli matrices
def pauli_matrices():
    """
    Gives list of the 4 single qubit Pauli matrices in the order I,X,Y,Z. 

    Returns: List of jnp array of all pauli_matrices
    """
    sigma_i = jnp.eye(2)
    sigma_x = jnp.array([[0, 1], [1, 0]])
    sigma_y = jnp.array([[0, -1j], [1j, 0]])
    sigma_z = jnp.array([[1, 0], [0, -1]])
    return [sigma_i, sigma_x, sigma_y, sigma_z]


def pauli_basis_strings(n_qubits):
    """
    Generates n qubit Pauli basis strings.

    Args:
        n_qubits (int): The number of qubits.
    Returns:
        List of strings representing the n qubit Pauli basis, e.g. ['II', 'IX', 'IY', 'IZ', 'XI', 'XX', 'XY', '....]
    """

    basis = ['I', 'X', 'Y', 'Z']
    return [''.join(p) for p in product(basis, repeat=n_qubits)]



def density_matrix_to_pauli_vector(density_matrix):
    """Takes a density matrix and converts it to a Pauli vector (Includes the Identity cofficient)
    
    Args:
        density_matrix (jnp.ndarray): The input density matrix to be converted.
    Returns:        
        jnp.ndarray: A vector representing the Pauli coefficients of the input density matrix.
    """

    n_qubits = int(jnp.log2(density_matrix.shape[0]))
    
    pauli_mats = pauli_matrices()

    # Initialize Pauli vector
    pauli_vector = jnp.zeros(4 ** n_qubits, dtype=complex)
    normalisation = jnp.sqrt(1/2**n_qubits)
    for pauli_string_idx, pauli_string in enumerate(pauli_basis_strings(n_qubits)):

        if n_qubits > 1:
            kron_prod = ft.reduce(jnp.kron, [pauli_mats['IXYZ'.index(p)]  for p in pauli_string])*normalisation

        else: kron_prod = [pauli_mats['IXYZ'.index(p)] for p in pauli_string][0]*normalisation

        pauli_vector = pauli_vector.at[pauli_string_idx].set(jnp.real(jnp.trace(kron_prod.dot(density_matrix))))
    return pauli_vector



def pauli_vector_to_density_matrix(pauli_vector,n_qubits=None):
    """ Takes a pauli vector and returns the corresponding density matrix (format w. Identity)
    Args:
        pauli_vector (jnp.ndarray): The input Pauli vector to be converted.
        n_qubits (int, optional): The number of qubits. If None, it
    
    Returns:
                jnp.ndarray: A density matrix corresponding to the input Pauli vector."""
    
    if n_qubits is None:
        n_qubits = int(round(jnp.log2(pauli_vector.shape[0]) / 2))
    v = pauli_vector[1:]
    norm = jnp.sqrt(jnp.dot(v.conjugate().T, v))
    jax.debug.print(
        "WARNING: Unphysical coherence vector! Norm = {n}",
        n=jax.lax.cond(norm > 1, lambda: norm, lambda: jnp.zeros_like(norm))
    )
    
    #n_qubits_vector = (jnp.log2(pauli_vector.shape[0]) / 2).astype(int)
    #if n_qubits_vector != n_qubits:
    #    print("specified number of qubits does not match the size of the vector")
    #if not (jnp.abs(pauli_vector[0]-  1/jnp.sqrt(2**n_qubits))< 1e-6):
    #    print("Not trace 1. Trace = ", pauli_vector[0] * n_qubits)
    pauli_mats = pauli_matrices()

    density_matrix = jnp.zeros((2 ** n_qubits, 2 ** n_qubits), dtype=complex)
    normalisation = jnp.sqrt(1/2**n_qubits)
    for pauli_string_idx, pauli_string in enumerate(pauli_basis_strings(n_qubits)):
        if n_qubits > 1:
            kron_prod = ft.reduce(jnp.kron, [pauli_mats['IXYZ'.index(p)] for p in pauli_string])*normalisation
        else: kron_prod = [pauli_mats['IXYZ'.index(p)] for p in pauli_string][0] * normalisation
        density_matrix +=  pauli_vector[pauli_string_idx] * kron_prod
    return density_matrix

def string_to_vector(pauli_string):
        """args: 
            pauli_string .... string or list of strings secifying e.g "ZX" or ["Z","X]
        returns:
            vector representation of that string """
        string_array = ['I', 'X', 'Y', 'Z']
        string_to_index = {string: index for index, string in enumerate(string_array)}
        pauli_b4 = [string_to_index[string] for string in pauli_string]
        index = 0
        for i,s in enumerate(pauli_b4[::-1]): 
            index += 4**i * s
        res = np.zeros(4**len(pauli_string))
        res[index] = 1
        return res






def calculate_steadystate_solve(Gate_list,param_list,timeit = False):
    """Given a description of a quantum circuit parameterized by params it returns the steady state Using solve (by far the quickest)

    Args:
        Gate_list (list): A list of gate objects, each containing information about the gate type, parameters, and location in the circuit.
        param_list (list): A list of parameters to be used for parameterized gates in the circuit. The parameters should be ordered according to the gates in the Gate_list.
        timeit (bool): If True, prints the time taken for circuit product calculation and matrix inversion. Default is False.     
    Returns:
        A vector representing the steady state of the quantum circuit defined by the Gate_list and param_list                      
    """

    #Go through all the channels backwards check if they are nonunital if nonunital add the 
    if timeit: t0 = time.time()
    i = 0
    gate0 = Gate_list[0] 
    n_params = gate0.n_params()
    if n_params != 0: 
        gate0.params = param_list[i:i+n_params]
        i += n_params
    Omega_prod,dj = gate0.rcircuit_gate()

    for gate in Gate_list[1:]: 
        n_params = gate.n_params()
        if n_params != 0: 
            gate.params = param_list[i:i+n_params]
            i += n_params
        Omegaj, cj = gate.rcircuit_gate()
        dj = Omegaj @ dj + cj 
        Omega_prod = Omegaj @ Omega_prod

    n_qubits = Gate_list[0].N_qubits
    mat = jnp.identity(int(4**n_qubits-1))-Omega_prod
    if timeit:  t1 = time.time()
    res = jnp.linalg.solve(mat,dj)
    if timeit:  
        t2 = time.time()
        print("times. Produict calc", t1-t0 ,"inverse", t2-t1)
    return res

def circuit_matrix(Gate_list,param_list):
    """
    Takes a list of gates w. specified locations and returns the parameterized transformation matrix corresponding to the circuit.
    
    Args:
        Gate_list (list): A list of gate objects, each containing information about the gate type, parameters, and location in the circuit.
        param_list (list): A list of parameters to be used for parameterized gates in the circuit. The parameters should be ordered according to the gates in the Gate_list.
        
    Returns:       
        A matrix representing the overall transformation of the quantum circuit defined by the Gate_list and param_list.
    """
    #go through all gates 
    i = 0 
    N_qubits = Gate_list[0].N_qubits
    circuit_product = jnp.identity(4**N_qubits)
    for gate in Gate_list:
        #parameterize the gates 
        n_params = gate.n_params()
        if n_params != 0:
            gate.params = param_list[i:i+n_params]
            i += n_params
        #Calculate the matrix product
        circuit_product = gate.circuit_gate() @ circuit_product
    return circuit_product
