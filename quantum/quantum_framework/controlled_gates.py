"""
controlled_gates.py

This module contains functions to append conditional unitary gates to a QuantumCircuit.
The conditional gate is constructed based on the provided adjacency matrix, where each
element determines the transformation when the control qubit is in the |1⟩ state.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate

def append_conditional_gates(adjacency_matrix, quantum_circuit):
    """
    Appends conditional controlled gates to the given quantum circuit.
    
    For each element a_ij of the adjacency_matrix, if a_ij is non-zero, a controlled
    unitary gate is created. The gate is defined by the 2x2 matrix:
    
        [[1 - a_ij, a_ij],
         [      1,      0]]
    
    This gate is applied with the control qubit at index j and target at index i.
    
    Parameters:
        adjacency_matrix (np.ndarray): A square numpy array with elements in [0, 1].
        quantum_circuit (QuantumCircuit): The circuit to which the gates will be appended.
    """
    num_rows, num_cols = adjacency_matrix.shape
    for i in range(num_rows):
        for j in range(num_cols):
            # Only apply a gate if the adjacency value is non-zero.
            if adjacency_matrix[i, j] != 0:
                # Define the unitary operation when the control qubit is in state |1⟩.
                matrix_if_1 = np.array([
                    [1 - adjacency_matrix[i, j], adjacency_matrix[i, j]],
                    [1, 0]
                ])
                gate_if_1 = UnitaryGate(matrix_if_1, label="U1")
                # Append the controlled version of the gate to the circuit.
                # Here, qubit j acts as the control and qubit i is the target.
                quantum_circuit.append(gate_if_1.control(1), [j, i])
