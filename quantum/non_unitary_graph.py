import numpy as np
from scipy.linalg import svd, qr

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt

# --- Utility Functions ---
s_values = []
def kronecker_product(matrices):
    """Compute the Kronecker product of a list of matrices."""
    result = np.eye(1)
    for matrix in matrices:
        result = np.kron(result, matrix)
    return result

def insert_non_unitary(matrix):
    """
    Embed a non-unitary operator into a larger unitary using a dilation.
    Uses SVD to compute a scaling factor and QR decomposition to complete the block.
    """
    U_, Sigma, Vh = svd(matrix)
    s2 = np.max(Sigma**2)
    s = 1 / np.sqrt(s2)
    s_values.append(s)
    Sigma_tilde = np.sqrt(np.maximum(0, 1 - s**2 * Sigma**2))
    C = U_ @ np.diag(Sigma_tilde) @ Vh

    B_tilde = np.random.rand(matrix.shape[0], matrix.shape[0])
    D_tilde = np.random.rand(matrix.shape[0], matrix.shape[0])
    U_tilde = np.block([[s * matrix, B_tilde],
                        [C, D_tilde]])
    UA, _ = qr(U_tilde)
    return UA

def quantum_circuit(num_qubits, non_unitary_operator, *states, add_measurements=True):
    """
    Constructs a quantum circuit that applies the dilated non-unitary operator.
    Creates a register for the main qubits and one for the ancilla.
    The main qubits are initialized using the provided state vectors.
    If add_measurements is True, it explicitly measures the registers.
    """
    main_qubits = QuantumRegister(num_qubits, 'main')
    ancilla_qubits = QuantumRegister(1, 'ancilla')
    main_clbits = ClassicalRegister(num_qubits, 'main_cl')
    ancilla_clbits = ClassicalRegister(1, 'ancilla_cl')

    qc = QuantumCircuit(main_qubits, ancilla_qubits, main_clbits, ancilla_clbits)
    
    # Initialize the main qubits (order: qubit 0, qubit 1)
    for i, state in enumerate(states):
        qc.initialize(state, i)
    
    # Note: ancilla is not explicitly initialized, so it remains in |0>.
    
    # Embed the non-unitary operator into a larger unitary.
    operator = insert_non_unitary(non_unitary_operator)
    gate = UnitaryGate(operator)
    # Apply the unitary to all qubits (main + ancilla)
    qc.append(gate, list(range(num_qubits + 1)))
    
    if add_measurements:
        qc.measure(main_qubits, main_clbits)
        qc.measure(ancilla_qubits, ancilla_clbits)
    
    return qc

def simulate_qc(qc, shots=1024):
    """Simulate the quantum circuit (with measurements) using Qiskit's Aer simulator."""
    simulator = Aer.get_backend('aer_simulator')
    compiled_circuit = transpile(qc, simulator)
    result = simulator.run(compiled_circuit, shots=shots).result()
    counts = result.get_counts()
    return counts

def simulate_statevector(qc):
    """Simulate the state vector of a circuit (with measurements removed)."""
    qc_no_meas = qc.remove_final_measurements(inplace=False)
    state = Statevector.from_instruction(qc_no_meas)
    return state.probabilities_dict()

# --- Main Diagnostic Loop ---

num_qubits = 2  # main qubits
shots = 1024

# Sweep parameter a from 0.2 to 1.
a_values = np.linspace(0.2, 1, 20)
measured_failure = np.zeros(len(a_values))
test_measurement = np.zeros(len(a_values))
sv_failure = np.zeros(len(a_values))  # statevector failure probability

# Also, record full counts (diagnostics) for a few values.
diagnostic_a = [0.2, 0.5, 0.8, 1.0]
diagnostic_results = {}

# Initialize with other initial states for main qubits

#initial_states = [[0,1],[1,0]] 
# This one works very well, but the s**2 underestimates it

#initial_states = [[np.sqrt(0.1),np.sqrt(0.9)],[1,0]]  
# This one is underestinated by both, but
# it is an issue of the expected failure, not so much of the plot

#initial_states = [[np.sqrt(0.1),np.sqrt(0.9)],[np.sqrt(0.9),np.sqrt(0.1)]] 
# The plot is the worst with this one

initial_states = [[1,0],[np.sqrt(0.9),np.sqrt(0.1)]] 
# This one is underestimated by both corrections
# but is the most underestimated by the 1/(np.sqrt(a) -1 ) correction

#initial_states = [[1,0], [1/np.sqrt(2), 1/np.sqrt(2)]]

for idx, a in enumerate(a_values):
    # Define the 2x2 identity and projectors.
    I = np.eye(2)
    # For the control qubit:
    P0 = np.array([[1, 0], [0, 0]])  # projector onto |0>
    P1 = np.array([[0, 0], [0, 1]])  # projector onto |1>
    
    # Define the non-unitary operator U (acting on the target).
    U = np.array([[np.sqrt(1 - a), 0],
                  [np.sqrt(a), 1]])
    
    # Construct the controlled operator.
    # In Qiskit's ordering (state vector is |main[1], main[0]>),
    # we want: if control (main[0]) is |0> then apply I on target (main[1]),
    # and if control is |1> then apply U on target.
    # Thus, we form:
    CU = np.kron(I, P0) + np.kron(U, P1)
    
    # Initialize the main qubits.
    # To activate the U branch, set:
    # - main qubit 0 (control) to |1> => [0, 1]
    # - main qubit 1 (target)  to |0> => [1, 0]
    qc = quantum_circuit(num_qubits, CU, initial_states[0], initial_states[1], add_measurements=True)
    
    # Run QASM simulation.
    counts = simulate_qc(qc, shots=shots)
    # Print full counts for diagnosis.
    print("For a =", a, "QASM counts:", counts)
    
    # In our printed keys the format is "<ancilla> <main>".
    # We take the failure branch as ancilla = 0 and main = 11.
    key = "0 11"
    if key in counts:
        measured_failure[idx] = counts[key] / shots
    else:
        measured_failure[idx] = 0

    key = "0 10"
    if key in counts:
        measured_failure[idx] += counts[key] / shots
        test_measurement[idx] = counts[key] / shots

    
    # Run statevector simulation for every a.
    qc_no_meas = quantum_circuit(num_qubits, CU, initial_states[0], initial_states[1], add_measurements=False)
    sv_probs = simulate_statevector(qc_no_meas)
    # In the statevector dictionary, keys appear without spaces.
    # We assume that "0 11" corresponds to "011": ancilla=0, main=11.
    sv_key = "011"
    if sv_key in sv_probs:
        sv_failure[idx] = sv_probs[sv_key]
    else:
        sv_failure[idx] = 0
    
    # Save diagnostic statevector data for selected a values.
    if np.any(np.isclose(a, diagnostic_a, atol=1e-2)):
        diagnostic_results[a] = sv_probs

# --- Plotting the Results ---


plt.figure(figsize=(10,6))

#expected_failure = a_values / (1 + np.sqrt(a_values))
# Another computation for the expected failure from the initial states

success = np.sqrt(1-a_values) * initial_states[1][0]
failure = np.sqrt(a_values) * initial_states[1][0] + initial_states[1][1]

# control qubit amplitudes
p,q = initial_states[0][0], initial_states[0][1]
print(f"Control = ({p})*|0> + ({q})*|1>")
# target qubit amplitudes
x,y = initial_states[1][0], initial_states[1][1]
print(f"Target = ({x})*|0> + ({y})*|1>")


# Non-allowed amplitudes
left = p**2 * x**2 +  q**2 * x**2 * (1-a_values) # |<0|psi>|**2 desired
right = p**2 * y**2 + q**2 * (a_values) # |<1|psi>|**2 desired

norm_term = left + right
expected_failure = right / norm_term

#expected_failure = 1 - (1-a_values)*(initial_states[1][0]**2)
print("Expected Failure")
plt.plot(a_values, expected_failure / (1 + np.sqrt(a_values)), 'k--', label='Expected Failure (a/(1+sqrt(a)))', alpha = 1, color ='blue')
plt.plot(a_values, measured_failure, 'o-', label='Measured Failure (QASM)', alpha= 0.5)
plt.plot(a_values, measured_failure * (1+np.sqrt(a_values)), 'o-', label = 'Corrected Measured Failure', alpha = 0.5, color = 'green')
plt.plot(a_values, sv_failure, 's-', label='Failure from Statevector', alpha = 0.5)
plt.plot(a_values, expected_failure, 'k--', label='Expected Failure (a)', color = 'green')
plt.xlabel('a')
plt.ylabel('Failure Probability (branch "0 11")')
plt.legend()
plt.title('Comparison of Failure Probability vs. Parameter a')
plt.grid(True)
plt.show()

# Plot the ratio of measured failure to expected failure
ratio = measured_failure / a_values
plt.figure(figsize=(10,6))
plt.plot(a_values, ratio, 'o-', label='Measured / Expected')
plt.xlabel('a')
plt.ylabel('Measured Failure / Expected (a)')
plt.title('Ratio of Measured Failure to Expected Value')
plt.legend()
plt.grid(True)
plt.show()

# --- Print Selected Diagnostics ---
print("\nSelected Diagnostic Data:")
for a_val, probs in diagnostic_results.items():
    print(f"a = {a_val:.3f} -> Statevector probabilities: {probs}")
    
print("\nComparison for Selected a Values:")
for test_index in np.arange(0,len(a_values),len(a_values)//5):
    print(f"a = {a_values[test_index]:.3f} -> QASM Failure = {measured_failure[test_index]:.4f}, "
          f"SV Failure = {sv_failure[test_index]:.4f}, Expected = {a_values[test_index]:.4f}")