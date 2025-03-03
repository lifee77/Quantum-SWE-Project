# dilation.py
import numpy as np
from scipy.linalg import svd, qr

def insert_non_unitary(matrix):
    """
    Embed a non-unitary operator into a larger unitary using dilation.
    Uses SVD to compute a scaling factor and QR decomposition to complete the block.
    
    Parameters:
        matrix (np.ndarray): The non-unitary operator (2^n x 2^n)
    
    Returns:
        UA (np.ndarray): The dilated unitary operator.
    """
    U_, Sigma, Vh = svd(matrix)
    s2 = np.max(Sigma**2)
    s = 1 / np.sqrt(s2)
    Sigma_tilde = np.sqrt(np.maximum(0, 1 - s**2 * Sigma**2))
    C = U_ @ np.diag(Sigma_tilde) @ Vh

    B_tilde = np.random.rand(matrix.shape[0], matrix.shape[0])
    D_tilde = np.random.rand(matrix.shape[0], matrix.shape[0])
    U_tilde = np.block([[s * matrix, B_tilde],
                        [C, D_tilde]])
    UA, _ = qr(U_tilde)
    return UA
