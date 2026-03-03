import warnings
import math
import jax.numpy as jnp
import scipy.linalg as la
from gates_jax import pauli_vector_to_density_matrix


def purity_cv(cv):
    """Calculates the purity from a given coherence vector
    

    Args:
        cv (ndarray): The coherence vector.

    Returns:
        float: The purity of the state.
    """
    v = jnp.sqrt(jnp.dot(cv[1:].conjugate().T,cv[1:]))
    d = jnp.sqrt(jnp.shape(cv)[0])
    return v**2 +1/d

###############################################################
# State manipulation.
###############################################################


def partial_trace(state, trace_systems, reverse=False):
    """
    Partial trace over subsystems of multi-partite matrix.

    Note that subsystems are ordered as rho012 = rho0(x)rho1(x)rho2.

    Args:
        state (matrix_like): a matrix NxN
        trace_systems (list(int)): a list of subsystems (starting from 0) to
                                  trace over.
        dimensions (list(int)): a list of the dimensions of the subsystems.
                                If this is not set it will assume all
                                subsystems are qubits.
        reverse (bool): ordering of systems in operator.
            If True system-0 is the right most system in tensor product.
            If False system-0 is the left most system in tensor product.

    Returns:
        matrix_like: A density matrix with the appropriate subsystems traced
            over.
    Raises:
        Exception: if input is not a multi-qubit state.
    """
    """if dimensions is None:  # compute dims if not specified
        num_qubits = int(jnp.log2(len(state)))
        dimensions = [2 for _ in range(num_qubits)]
        if len(state) != 2**num_qubits:
            raise Exception("Input is not a multi-qubit state, "
                            "specify input state dims")
    else:
        dimensions = list(dimensions)"""
    
    num_qubits = int(round(math.log2(state.shape[0])))
    dimensions = [2 for _ in range(num_qubits)]
    if isinstance(trace_systems, int):
        trace_systems = [trace_systems]
    else:  # reverse sort trace sys
        trace_systems = sorted(trace_systems, reverse=True)

    # trace out subsystems
    if state.ndim == 1:
        # optimized partial trace for input state vector
        return __partial_trace_vec(state, trace_systems, dimensions, reverse)
    # standard partial trace for input density matrix
    return __partial_trace_mat(state, trace_systems, dimensions, reverse)


def __partial_trace_vec(vec, trace_systems, dimensions, reverse=True):
    """
    Partial trace over subsystems of multi-partite vector.

    Args:
        vec (vector_like): complex vector N
        trace_systems (list(int)): a list of subsystems (starting from 0) to
                                  trace over.
        dimensions (list(int)): a list of the dimensions of the subsystems.
                                If this is not set it will assume all
                                subsystems are qubits.
        reverse (bool): ordering of systems in operator.
            If True system-0 is the right most system in tensor product.
            If False system-0 is the left most system in tensor product.

    Returns:
        ndarray: A density matrix with the appropriate subsystems traced over.
    """

    # trace sys positions
    if reverse:
        dimensions = dimensions[::-1]
        trace_systems = len(dimensions) - 1 - jnp.array(trace_systems)

    rho = vec.reshape(dimensions)
    rho = jnp.tensordot(rho, rho.conj(), axes=(trace_systems, trace_systems))
    d = jnp.sqrt(jnp.product(rho.shape)).astype(int)

    return rho.reshape(d, d)


def __partial_trace_mat(mat, trace_systems, dimensions, reverse=True):
    """
    Partial trace over subsystems of multi-partite matrix.

    Note that subsystems are ordered as rho012 = rho0(x)rho1(x)rho2.

    Args:
        mat (matrix_like): a matrix NxN.
        trace_systems (list(int)): a list of subsystems (starting from 0) to
                                  trace over.
        dimensions (list(int)): a list of the dimensions of the subsystems.
                                If this is not set it will assume all
                                subsystems are qubits.
        reverse (bool): ordering of systems in operator.
            If True system-0 is the right most system in tensor product.
            If False system-0 is the left most system in tensor product.

    Returns:
        ndarray: A density matrix with the appropriate subsystems traced over.
    """

    trace_systems = sorted(trace_systems, reverse=True)
    for j in trace_systems:
        # Partition subsystem dimensions
        dimension_trace = int(dimensions[j])  # traced out system
        if reverse:
            left_dimensions = dimensions[j + 1:]
            right_dimensions = dimensions[:j]
            dimensions = right_dimensions + left_dimensions
        else:
            left_dimensions = dimensions[:j]
            right_dimensions = dimensions[j + 1:]
            dimensions = left_dimensions + right_dimensions
        # Contract remaining dimensions
        dimension_left = math.prod(left_dimensions)
        dimension_right = math.prod(right_dimensions)

        # Reshape input array into tri-partite system with system to be
        # traced as the middle index
        mat = mat.reshape([
            dimension_left, dimension_trace, dimension_right, dimension_left,
            dimension_trace, dimension_right
        ])
        # trace out the middle system and reshape back to a matrix
        mat = mat.trace(axis1=1, axis2=4).reshape(
        dimension_left * dimension_right,
        dimension_left * dimension_right
        )
    return mat

def fidelity(A,B):
    """Fidelity between two quantum states, states can be given as density matrices or as coherence vectors.

    Args: 
        A (ndarray): Density matrix or coherence vector of state A
        B (ndarray): Density matrix or coherence vector of state B
    Returns:
        float: Fidelity between states A and B
    """
    if A.ndim == 1:
        A = pauli_vector_to_density_matrix(A)
    if B.ndim == 1:
        B = pauli_vector_to_density_matrix(B)
    # Ensure input is jax.numpy array
    A = jnp.asarray(A)
    B = jnp.asarray(B)
    # Compute sqrtm(A) using scipy (since jax does not have sqrtm)
    sqrtmA = la.sqrtm(jnp.array(A))
    # Compute sqrtmA * B * sqrtmA
    prod = sqrtmA @ jnp.array(B) @ sqrtmA
    # Compute eigenvalues
    eigvals = la.eigvalsh(prod)
    # Truncate negative eigenvalues (numerical errors)
    eigvals = jnp.clip(eigvals, 0, None)
    # Fidelity is the squared sum of square roots of eigenvalues
    return float(jnp.real(jnp.sum(jnp.sqrt(eigvals))))