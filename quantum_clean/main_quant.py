"""
Module main_quant.py

Centralizes the operations from other modules. Makes the experiments
and compares them with the expected results from the operations. 
Makes histograms and plots the results. Some initial superpositions are
displayed here.
"""
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer
from scipy.linalg import svd, qr
from qiskit.circuit.library import UnitaryGate
import run_quantum_circuit as run_qc
import non_unitary_ops as nu_ops
from qiskit.visualization import plot_histogram

def run_circuit(a, psi0, psi1, shots=2000):
    """
    Build a circuit for the 2-main-qubits + 1-ancilla scenario:
      - main[0] = control qubit, initialized to psi0
      - main[1] = target qubit, initialized to psi1
      - anc in |0>
      - non-unitary M_ij depends on 'a'
    Return the probability that qubit-1 is measured = |1>.
    """
    # Create quantum and classical registers
    qreg_main = QuantumRegister(2, 'main')
    qreg_anc = QuantumRegister(1, 'anc')
    creg = ClassicalRegister(3, 'c')  # measure 3 qubits
    qc = QuantumCircuit(qreg_main, qreg_anc, creg)

    # Initialize qubits
    qc.initialize(psi0, qreg_main[0])  
    qc.initialize(psi1, qreg_main[1])  

    # Construct the 2x2 operator
    U_ij = np.array([
        [np.sqrt(1 - a), 0],
        [np.sqrt(a),     1]
    ], dtype=complex)

    # Create the controlled-operation matrix
    I2 = np.eye(2, dtype=complex)
    P0 = np.array([[1,0],[0,0]], dtype=complex)
    P1 = np.array([[0,0],[0,1]], dtype=complex)
    M_ij = np.kron(I2, P0) + np.kron(U_ij, P1)

    # Embed into 3-qubit unitary
    big_unitary = nu_ops.embed_non_unitary(M_ij)
    gate = UnitaryGate(big_unitary, label="M'_ij")

    # Append to the circuit on qubits [0,1,2]
    qc.append(gate, [0, 1, 2])

    # Measure all qubits
    qc.measure(qreg_anc, 2)
    qc.measure(qreg_main[0], 0)
    qc.measure(qreg_main[1], 1)

    # Run simulation
    backend = Aer.get_backend('qasm_simulator')
    compiled = transpile(qc, backend)
    job = backend.run(compiled, shots=shots)
    result = job.result()
    counts = result.get_counts()

    # Probability that qubit1 = |1> AND ancilla = |0>
    p_q1_1 = sum(c for key, c in counts.items() if key[1] == '1' and key[0] == '0')/shots
    return p_q1_1

# Set up the initial states
# "qubit0" = control, "qubit1" = target
# Each qubit is a 2D vector [amp|0>, amp|1>]

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
    "|1,0>": init_10,
    "|1,1>": init_11,
    "|+,0>": init_plus0,
    "|+,+>": init_plus_plus
}

# AER Simulations and Expected
a_values = np.linspace(0, 1, 11)
results = {}  # store p(q1=1) for each state vs. a
scaled_results = {}
theoretical_results = {}
shots = 2000

for label, (psi0, psi1) in initial_states.items():
    probs = []
    s_probs = []
    t_probs = []
    for a in a_values:
        # Get measurement results
        p = run_circuit(a, psi0, psi1, shots=shots)
        probs.append(p)
        
        # Apply scaling factor
        scaling_factor = nu_ops.get_rescaling_factor(a)
        s_probs.append(p * scaling_factor)
        
        # Calculate theoretical expectation
        t_probs.append(nu_ops.expected_failure(a, psi0, psi1))
    
    results[label] = probs
    scaled_results[label] = s_probs
    theoretical_results[label] = t_probs

# Create a 1x2 grid of subplots (side-by-side)
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), sharex=True)

# (A) Left subplot: Measured Probability
# --------------------------------------------------------------
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

# Print theoretical values for |1,0>
print("Theoretical failure probabilities for |1,0>:", theoretical_results["|1,0>"])