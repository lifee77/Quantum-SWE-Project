# main.py
import numpy as np
from dilation import insert_non_unitary
from controlled import projector_0, projector_1, construct_controlled_operator
from simulation import build_quantum_circuit, run_simulation, run_statevector_simulation
from visualization import plot_failure

# Example: Build a circuit for 2 qubits with a controlled operator.
num_qubits = 2
shots = 1024

# Define parameter 'a' and the non-unitary operator U.
a = 0.3
I = np.eye(2)
U = np.array([[np.sqrt(1-a), 0],
              [np.sqrt(a), 1]])
# Construct a controlled operator using functions from controlled.py
# For simplicity, here we assume the control is on qubit 0.
# (You can modify this function to account for your ordering and desired logic.)
CU = np.kron(I, projector_0()) + np.kron(U, projector_1())

# Define states to initialize the main qubits.
# For example, to activate the U branch, set:
states = [[0, 1], [1, 0]]  # qubit0 in |1>, qubit1 in |0>

# Build the quantum circuit
qc = build_quantum_circuit(num_qubits, CU, states, add_measurements=True)

# Run simulation
counts = run_simulation(qc, shots=shots)
print("QASM Counts:", counts)

# Optionally, run statevector simulation
sv = run_statevector_simulation(qc)
print("Statevector probabilities:", sv)

# Suppose you sweep over a range of 'a' values.
a_values = np.linspace(0.2, 1, 50)
measured_failure = []
expected_failure = []

for a in a_values:
    U = np.array([[np.sqrt(1-a), 0],
                  [np.sqrt(a), 1]])
    CU = np.kron(I, projector_0()) + np.kron(U, projector_1())
    qc = build_quantum_circuit(num_qubits, CU, states, add_measurements=True)
    counts = run_simulation(qc, shots=shots)
    key = "0 11"  # Adjust this key based on your output format.
    m_fail = counts.get(key, 0) / shots
    measured_failure.append(m_fail)
    
    # Expected probability after dilation scaling:
    # If the scaling factor s = 1/sqrt(1+sqrt(a)), then expected is a/(1+sqrt(a))
    expected_failure.append(a/(1+np.sqrt(a)))

measured_failure = np.array(measured_failure)
expected_failure = np.array(expected_failure)

plot_failure(a_values, measured_failure, expected_failure, title='Failure Probability vs a')
