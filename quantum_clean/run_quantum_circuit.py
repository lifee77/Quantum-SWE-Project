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
import os


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

def measure_failure(a, psi0, psi1, backend, shots=1024, filtered= False):
    """
    Measures the failure probability of a quantum circuit.
    Inputs:
        - a:       The real parameter in [0,1].
        - psi0:    Amplitude for the control qubit |j>.
        - psi1:    Amplitude for the target qubit |i>.
        - backend:  The quantum backend to run the circuit on.
        - shots:   Number of shots to run.
        - filtered: If True, return also the filtered results
    Outputs:
        - p:      Probability of measuring |1> on the target qubit.

    """
    results = run_circuit(a, psi0, psi1, backend, shots)
    filtered_results = filter_ancilla(results)
    counts = counts_failure(filtered_results)
    if not filtered:
        return counts / shots
    else:
        return counts / shots, filtered_results


def create_state_histogram(state_counts, n_nodes, output_dir="classical/results/"):
    """
    Create a histogram plot of state occurrences.

    ***Repeated function somewhere in the repo!!!
    
    Args:
        state_counts (dict): Dictionary of states and their counts
        n_nodes (int): Number of nodes in the system
        output_dir (str): Directory to save the plot
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort states for better visualization
    sorted_states = sorted(state_counts.items(), key=lambda x: int(x[0], 2))
    states, counts = zip(*sorted_states) if sorted_states else ([], [])
    
    # Calculate probabilities for each state
    total_runs = sum(counts)
    probabilities = [count / total_runs for count in counts]
    
    # Create figure and axes
    plt.figure(figsize=(12, 6))
    
    # Plot histogram with blue bars
    bars = plt.bar(range(len(states)), probabilities, color='royalblue', alpha=0.7)
    
    # Add text annotations inside bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height/2,
                 f'{count}', ha='center', va='center', color='white', fontweight='bold')
    
    # Set labels and title
    plt.title(f'State Distribution for Classical System with {n_nodes} Nodes')
    plt.xlabel('State')
    plt.ylabel('Probability')
    
    # Set x-tick labels to show binary states
    plt.xticks(range(len(states)), states, rotation=45)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f"{output_dir}classical_simulation_n{n_nodes}.png")
    plt.show() # This was not in the original function
    plt.close()