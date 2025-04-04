"""
Module main_quant.py

Centralizes the operations from other modules. Makes the experiments
and commpares them with the expected results from the operations. 
Makes histograms and plots the results. Some initial superpositions are
displayed here
"""
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer
from scipy.linalg import svd, qr
from qiskit.circuit.library import UnitaryGate
#from run_quantum_circuit import run_circuit
import run_quantum_circuit as run_qc
import non_unitary_ops as nu_ops
from qiskit.visualization import plot_histogram


# 1) qubit0=|1>, qubit1=|0>
init_10 = (np.array([0, 1], dtype=complex),  # qubit0
           np.array([1, 0], dtype=complex))  # qubit1

# 2) qubit0=|1>, qubit1=|1>
init_11 = (np.array([0, 1], dtype=complex),
           np.array([0, 1], dtype=complex))

# 3) qubit0 = (|0>+|1>)/sqrt(2), qubit1=|0>
init_plus0 = (np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex),
              np.array([1, 0], dtype=complex))

# 4) qubit0 = (|0>+|1>)/sqrt(2), qubit1=(|0>+|1>)/sqrt(2)
init_plus_plus = (
    np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex),
    np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
)

# Put them in a dictionary for easy looping/plotting:
initial_states = {
    "|0,1>": init_10,
    "|1,1>": init_11,
    "|0,+_0>": init_plus0,
    "|+_1,+_0>": init_plus_plus
}

# AER Simulations and Expected

a_values = np.linspace(0, 1, 11)
results = {}  # store p(q1=1) for each state vs. a
scaled_results = {}
theoretical_results = {}
shots = 2000
backend = Aer.get_backend('qasm_simulator')

for label, (psi0, psi1) in initial_states.items():
    probs = []
    s_probs = []
    t_probs = []
    for a in a_values:
        p = run_qc.measure_failure(a,psi0, psi1, backend, shots=shots)
        probs.append(p)

        #scaled
        s_probs.append( p * (np.sqrt(a) + 1) ) 

        # theoretical 
        t_probs.append(nu_ops.expected_failure(a, psi0, psi1))
    # title = str(psi0) + str(psi1)
    # plt.title(title)
    # plt.bar(counts.keys(), counts.values())
    # plt.title(psi0, psi1)
    # plt.show();
    results[label] = probs
    scaled_results[label] = s_probs
    theoretical_results[label] = t_probs

# Create a 1x2 grid of subplots (side-by-side)
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), sharex=True)

# (A) Left subplot: Measured Probability
# --------------------------------------------------------------
# We'll allow Matplotlib to pick colors from the default color cycle.
# For consistency, let’s store the colors in a list so we can re-use them in the second subplot.
color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
labels_list = list(initial_states.keys())  # So we can index them

for idx, label in enumerate(labels_list):
    ax1.plot(a_values,
             results[label],
             marker='o',
             linestyle='-',
             color=color_cycle[idx % len(color_cycle)],
             label=label)

ax1.set_xlabel(r"$a$", fontsize=12)
ax1.set_ylabel(r"Measured Probability $(q_1=|1\rangle)$", fontsize=12)
ax1.set_title("Measured Probability vs. $a$", fontsize=13)
ax1.legend(loc="best")
ax1.grid(True)

# (B) Right subplot: Scaled Probability + Theory
# --------------------------------------------------------------
for idx, label in enumerate(labels_list):
    color = color_cycle[idx % len(color_cycle)]
    
    # Plot scaled measurement
    ax2.plot(a_values,
             scaled_results[label],
             marker='o',
             linestyle='-',
             color=color,
             label=label + " (measured)")
    
    # Plot theory
    ax2.plot(a_values,
             theoretical_results[label],
             marker='x',
             linestyle='--',
             color=color,
             label=label + " (theory)")

ax2.set_xlabel(r"$a$", fontsize=12)
ax2.set_ylabel(r"Scaled Probability: $p \times (\sqrt{a} + 1)$", fontsize=12)
ax2.set_title("Scaled Probability vs. $a$", fontsize=13)
ax2.legend(loc="best")
ax2.grid(True)

plt.tight_layout()
plt.show()

print(theoretical_results["|1,0>"])

# Quantum Backend running the circuit