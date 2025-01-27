import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate

def append_conditional_gates(adjacency_matrix,quantum_circuit):
    for i in range(adjacency_matrix.shape[0]):
        for j in range(adjacency_matrix.shape[1]):
            #matrix if s_j is 1
            matrix_if_1 = np.array([[1-adjacency_matrix[i,j], adjacency_matrix[i,j]], [1,0]])
            #matrix if s_j is 0
            #matrix_if_0 = np.array([[1,0],[0,1]])

            gate_if_1 = UnitaryGate(matrix_if_1, label="U1")
            #gate_if_0 = UnitaryGate(matrix_if_0, label="U0")

            quantum_circuit.append(gate_if_1.control(1), [j,i])
            #quantum_circuit.append(gate_if_0.control(0), [j,i])
            #would be basically not doing anything, so we leave it




    

