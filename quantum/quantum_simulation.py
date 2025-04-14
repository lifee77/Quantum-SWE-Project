#!/usr/bin/env python3
# quantum_simulation.py

import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile
from qiskit.circuit.library import RYGate
from qiskit.circuit.library import UnitaryGate

def main():
    """
    Translation from the notebook into python code
    """
    # -------------
    # Section 1: Basic Configuration
    # -------------
    risk_vector = np.array([0.2, 0.3, 0.5])  # Example risk/probability inputs
    adjacency_matrix = np.array([
        [0,    0.10, 0.13],
        [0.20, 0,    0.50],
        [0.17, 0.30, 0   ]
    ])

    num_qubits = len(risk_vector)
    num_ancillas = num_qubits
    
    # Quantum + Classical Registers
    main_qubits = QuantumRegister(num_qubits, 'main')
    ancilla_qubits = QuantumRegister(num_ancillas, 'ancilla')
    main_clbits = ClassicalRegister(num_qubits, 'main_cl')
    ancilla_clbits = ClassicalRegister(num_ancillas, 'ancilla_cl')
    
    # Create the circuit
    qc = QuantumCircuit(main_qubits, ancilla_qubits, main_clbits, ancilla_clbits)
    
    # -------------
    # Section 2: Risk-based Rotations
    # -------------
    # For each qubit, rotate by an angle corresponding to the "risk" r_i
    #   angle θ = 2 * arcsin(sqrt(r_i))
    for idx, risk_val in enumerate(risk_vector):
        theta_ri = 2 * np.arcsin(np.sqrt(risk_val))
        qc.ry(theta_ri, main_qubits[idx])
    
    # -------------
    # Section 3: Entangle Qubits with Ancillas (CNOT)
    # -------------
    #  control = main_qubit[i], target = ancilla_qubit[i]
    for i in range(num_qubits):
        qc.cx(main_qubits[i], ancilla_qubits[i])
    
    # -------------
    # Section 4: Collapse Original Qubits by Measurement
    # -------------
    qc.measure(main_qubits, main_clbits)
    
    # -------------
    # Section 5: Add a Barrier Before Applying Adjacency-based Rotations
    # -------------
    qc.barrier()
    
    # -------------
    # Section 6: Controlled Rotations Based on Adjacency Matrix
    # -------------
    # For each adjacency_matrix[i][j] > 0, we apply a controlled rotation
    #   angle θ_ij = 2 * arcsin(sqrt(a_ij))
    # The control is the ancillas i, j and the target is main_qubit i (example approach).
    # Note: This is a triple-controlled scenario – adjust as needed for your model.
    for i in range(len(adjacency_matrix)):
        for j in range(len(adjacency_matrix[i])):
            a_ij = adjacency_matrix[i][j]
            if a_ij != 0:
                theta_ij = 2 * np.arcsin(np.sqrt(a_ij))
                # Build the base single-qubit rotation gate
                base_rotation = RYGate(theta_ij)
                # Now make it a 2-controlled gate
                controlled_rotation = base_rotation.control(num_ctrl_qubits=2)
                # Append to the circuit:
                #   controls = ancilla_qubits[i] & ancilla_qubits[j],
                #   target  = main_qubits[i].
                qc.append(controlled_rotation, [ancilla_qubits[i], ancilla_qubits[j], main_qubits[i]])
    
    # -------------
    # Section 7: Final Measurements
    # -------------
    qc.measure(ancilla_qubits, ancilla_clbits)
    
    # -------------
    # Section 8: Output or Visualization
    # -------------
    # In a Python script, you can print the circuit as text:
    print("\nFinal Quantum Circuit:\n")
    print(qc.draw(output='text'))
    
    # If you want to transpile (optimize) or run on a simulator/back-end:
    # from qiskit import Aer
    # from qiskit import transpile
    # sim = Aer.get_backend('qasm_simulator')
    # compiled_circuit = transpile(qc, sim)
    # job = sim.run(compiled_circuit, shots=1024)
    # result = job.result()
    # counts = result.get_counts()
    # print(counts)

if __name__ == "__main__":
    main()
