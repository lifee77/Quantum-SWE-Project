# controlled.py
import numpy as np

def construct_controlled_operator(U, control_proj, target_proj, I):
    """
    Construct a controlled operator for a 2-qubit system.
    
    Parameters:
        U (np.ndarray): The non-unitary operator (acting on the target).
        control_proj (np.ndarray): The projector for the control qubit when active (e.g., P1).
        target_proj (np.ndarray): The operator for the target when control is inactive (e.g., I).
        I (np.ndarray): The identity matrix (2x2).
    
    Returns:
        CU (np.ndarray): The controlled operator (4x4 matrix).
    """
    # When control is 0: apply identity on target.
    CU0 = np.kron(I, (np.eye(2) - control_proj))
    # When control is 1: apply U on target.
    CU1 = np.kron(U, control_proj)
    return CU0 + CU1

# Example projectors:
def projector_0():
    return np.array([[1, 0], [0, 0]])

def projector_1():
    return np.array([[0, 0], [0, 1]])
