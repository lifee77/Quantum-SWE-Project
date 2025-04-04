"""
Module: run_quantum_circuit.py

This module runs a quantum circuit using a specified backend and number of shots.

"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer
from scipy.linalg import svd, qr
from qiskit.circuit.library import UnitaryGate  
import non_unitary_ops as nu_ops

def run_circuit(a, psi0, psi1, backend, shots=1024):
    """
    Runs a single quantum circuit with a specified backend
    Inputs:
        - a:       The real parameter in [0,1].
        - psi0:    Amplitude for the control qubit |j>.
        - psi1:    Amplitude for the target qubit |i>.
        - backend:  The quantum backend to run the circuit on.
        - shots:   Number of shots to run.
    Outputs:
        - results: The counts of the measurement results. 
    """

    qc = nu_ops.build_qc(a, psi0, psi1)
    compiled = transpile(qc, backend)
    job = backend.run(compiled, shots=shots)
    result = job.result()

    return result.get_counts()

def filter_ancilla(result):
    """
    Obtains the counts of the results when the ancilla is |0>
    """
    filtered_results = {}

    for key, count in result.items():
        if key[0] == '0':
            filtered_results[key[1:]] = count
    
    return filtered_results

def counts_failure(filtered_results):
    """
    Obtains the counts of measuring |1> on the target qubit
    """
    return sum(c for key, c in filtered_results.items() if key[0]=='1')

def measure_failure(a, psi0, psi1, backend, shots=1024):
    """
    Measures the failure probability of a quantum circuit.
    Inputs:
        - a:       The real parameter in [0,1].
        - psi0:    Amplitude for the control qubit |j>.
        - psi1:    Amplitude for the target qubit |i>.
        - backend:  The quantum backend to run the circuit on.
        - shots:   Number of shots to run.
    Outputs:
        - p:      Probability of measuring |1> on the target qubit.
    """
    results = run_circuit(a, psi0, psi1, backend, shots)
    filtered_results = filter_ancilla(results)
    counts = counts_failure(filtered_results)

    return counts / shots