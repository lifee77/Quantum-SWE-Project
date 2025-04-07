"""
Module non_unitary_ops.py

Contains functions to construct and analyze non-unitary operations
for quantum systems, particularly focusing on controlled non-unitary operations
for cascade failure simulations.
"""

import numpy as np
from scipy.linalg import svd, qr

def controlled_operator(a):
    """
    Returns M_{ij} = |0><0|_j x I_i + |1><1|_j x U_{ij},
    where U_{ij} = [[sqrt(1 - a), 0], [sqrt(a), 1]].
    
    Args:
        a (float): Parameter representing failure probability in [0,1]
        
    Returns:
        numpy.ndarray: 4x4 matrix representing the controlled non-unitary operation
    """
    # Projectors on the control qubit j
    P0_j = np.array([[1, 0], [0, 0]])  # |0><0|
    P1_j = np.array([[0, 0], [0, 1]])  # |1><1|
    
    # 2x2 operator on the target qubit i
    Uij = np.array([
        [np.sqrt(1 - a), 0],
        [np.sqrt(a),     1]
    ])
    
    # Construct the 4x4 operator: |0><0|_j ⊗ I_i + |1><1|_j ⊗ U_{ij}
    return np.kron(P0_j, np.eye(2)) + np.kron(P1_j, Uij)

def expected_failure(a, j_init, i_init):
    """
    Calculate the theoretical probability of failure for the target qubit.
    
    Args:
        a (float): Parameter representing failure probability in [0,1]
        j_init (numpy.ndarray): Initial state of the control qubit
        i_init (numpy.ndarray): Initial state of the target qubit
    
    Returns:
        float: Probability that the target qubit is in state |1>
    """
    # Construct the 4D initial state vector in the order [|0,0>, |0,1>, |1,0>, |1,1>]
    initial_state = np.kron(j_init, i_init)  # shape (4,)

    # Apply the controlled operator
    Mij = controlled_operator(a)
    final_state = Mij @ initial_state

    # Compute the probability that the target qubit is |1>
    p_01 = np.abs(final_state[1])**2  # probability of |0,1>
    p_11 = np.abs(final_state[3])**2  # probability of |1,1>
    
    # Return total probability of target being |1> (no normalization needed)
    return p_01 + p_11

def embed_non_unitary(matrix):
    """
    Embed a non-unitary operator (matrix) into a larger unitary operator.
    
    Args:
        matrix (numpy.ndarray): Non-unitary matrix to be embedded
        
    Returns:
        numpy.ndarray: Unitary matrix that embeds the input matrix
    """
    # Perform SVD
    U_, Sigma, Vh = svd(matrix)
    
    # Scale factor = 1 / sqrt(max singular value^2)
    s2 = np.max(Sigma**2)
    s = 1/np.sqrt(s2)

    # Build the "C" block
    Sigma_tilde = np.sqrt(np.maximum(0, 1 - s**2 * Sigma**2))
    C = U_ @ np.diag(Sigma_tilde) @ Vh

    # Get dimensions and create random blocks
    size = matrix.shape[0]
    B_tilde = np.random.rand(size, size)
    D_tilde = np.random.rand(size, size)

    # Construct the block matrix
    U_tilde = np.block([
        [s * matrix,  B_tilde],
        [      C,    D_tilde]
    ])
    
    # Convert to a unitary via QR decomposition
    UA, _ = qr(U_tilde)
    return UA

def get_rescaling_factor(a):
    """
    Calculate the rescaling factor needed to convert measured probabilities
    to theoretical probabilities.
    
    Args:
        a (float): Parameter representing failure probability in [0,1]
        
    Returns:
        float: Rescaling factor (1 + sqrt(a))
    """
    # The rescaling factor is 1/s², where s is the scaling used in the embedding
    return 1 + np.sqrt(a)
