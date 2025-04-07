"""
Module: non_unitary_ops.py

This module implements controlled non-unitary operations running a quantum circuit
with the (scaled) effect of the following operator:

    M_{ij} = |0⟩⟨0|_j ⊗ I_i + |1⟩⟨1|_j ⊗ U_{ij}

where the non-unitary operation is defined as:
    
    U_{ij} = [[sqrt(1 - a_ij), 0],
              [sqrt(a_ij), 1]]

with a_ij being the probability that system i fails given system j has failed.
The current implementation supports a 2-qubit system.

functions: get_non_unitary, embed_non_unitary, build_qc, expected failure
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer
from scipy.linalg import svd, qr
from qiskit.circuit.library import UnitaryGate  


def get_non_unitary(a):
    """
    Returns the 4x4 non-unitary operator M_ij
    Inputs:
        - a:       The real parameter in [0,1].
    Outputs:
        - M_ij:    The non-unitary operator as a 4x4 numpy array.
    """

    # Construct the 2x2 operator
    #   U_ij = |0><0| + sqrt(a)|1><0| + ...
    U_ij = np.array([
        [np.sqrt(1 - a), 0],
        [np.sqrt(a),     1]
    ], dtype=complex)

    # We only apply this to qubit-1 if it's in |1>, so we do:
    #   M_ij = I \otimes P0 + U_ij \otimes P1
    I2 = np.eye(2, dtype=complex)
    P0 = np.array([[1,0],[0,0]], dtype=complex)
    P1 = np.array([[0,0],[0,1]], dtype=complex)
    return np.kron(P0,I2) + np.kron(P1,U_ij)

def embed_non_unitary(matrix):
    """
    Embed a non-unitary operator (matrix) into a larger 2N x 2N unitary.
    Uses SVD -> scale factor s -> build block matrix -> QR decomposition.

    Inputs:
        matrix: as a NxN numpy array
    Outputs:
        UA:     The 2N x 2N unitary matrix.
    """
    U_, Sigma, Vh = svd(matrix)
    # Scale = 1 / sqrt(max singular value^2)
    s2 = np.max(Sigma**2)
    s = 1/np.sqrt(s2)

    # Build the "C" block
    Sigma_tilde = np.sqrt(np.maximum(0, 1 - s**2 * Sigma**2))
    C = U_ @ np.diag(Sigma_tilde) @ Vh

    size = matrix.shape[0]
    B_tilde = np.random.rand(size, size)
    D_tilde = np.random.rand(size, size)

    U_tilde = np.block([
        [s * matrix,  B_tilde],
        [      C,    D_tilde]
    ])
    # Convert to a unitary by QR
    UA, _ = qr(U_tilde)
    return UA

def expected_failure(a, j_init, i_init):
    """
    Given:
      - a:       The real parameter in [0,1].
      - j_init:  A length-2 array of amplitudes for the control qubit |j>.
      - i_init:  A length-2 array of amplitudes for the target qubit |i>.

    1) Builds the full initial state |j> ⊗ |i> in a 4D vector.
    2) Applies the 4x4 operator M_{ij}(a).
    3) Returns the probability that the final state
       is in the subspace with the target qubit = |1> 
       (i.e., final basis states |0,1> and |1,1>).
    """
    
    # -- 1) Construct the 4D initial state vector in the order [|0,0>, |0,1>, |1,0>, |1,1>] --
    #    If j_init = [α, β] and i_init = [γ, δ], the tensor product is:
    #       |ψ_init> = αγ |0,0> + αδ |0,1> + βγ |1,0> + βδ |1,1>.
    initial_state = np.kron(j_init, i_init)  # shape (4,)

    # -- 2) Compute final state after applying M_{ij}(a) --
    Mij = get_non_unitary(a)
    final_state = Mij @ initial_state

    # -- 3) Compute the normalized probability that the target qubit is |1> --
    p_01 = np.abs(final_state[1])**2  # amplitude of |0,1>
    p_11 = np.abs(final_state[3])**2  # amplitude of |1,1>
    norm = np.sum(np.abs(final_state)**2)

    #return (p_01 + p_11) / norm
    return (p_01 + p_11)/ norm # no need to normalize?

def build_qc(a, psi0, psi1):
    """
    Build a quantum circuit with the controlled non-unitary operator.

    Inputs:
        - a:       The real parameter in [0,1].
        - psi0:    A length-2 array of amplitudes for the control qubit |j>.
        - psi1:    A length-2 array of amplitudes for the target qubit |i>.
    Returns:
        - qc:      The quantum circuit object.
    """

     # Create quantum and classical registers
    qreg_main = QuantumRegister(2, 'main')
    qreg_anc = QuantumRegister(1, 'anc')
    creg = ClassicalRegister(3, 'c')  # measure 3 qubits
    qc = QuantumCircuit(qreg_main, qreg_anc, creg)

    # Initialize qubits
    #   psi0 is a 2D array (amps for qubit0)
    #   psi1 is a 2D array (amps for qubit1)
    qc.initialize(psi0, qreg_main[0])  
    qc.initialize(psi1, qreg_main[1])  

    # Construct the non-unitary 2x2 operator
    M_ij = get_non_unitary(a)
    
    # Embed into 3-qubit unitary
    big_unitary = embed_non_unitary(M_ij)
    gate = UnitaryGate(big_unitary, label="M'_ij")

    # Append to the circuit on qubits [0,1,2]
    qc.append(gate, [0, 1, 2])

    # Measure qubit0 -> cbit0, qubit1-> cbit1, anc-> cbit2
    qc.measure(qreg_anc,     2)
    qc.measure(qreg_main[0], 0)
    qc.measure(qreg_main[1], 1)

    return qc
