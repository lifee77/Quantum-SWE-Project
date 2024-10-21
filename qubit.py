from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_bloch_multivector, plot_histogram
import matplotlib.pyplot as plt

# Create a Quantum Circuit with 1 qubit
qc = QuantumCircuit(1)

# Initial state of the qubit (|0⟩)
print("Initial state of the qubit:")
qc.draw(output='mpl')
plt.show()

# Apply Hadamard gate to create a state of superposition
qc.h(0)
print("State of superposition:")
qc.draw(output='mpl')
plt.show()

# Simulate the circuit
backend = Aer.get_backend('statevector_simulator')
result = execute(qc, backend).result()
statevector = result.get_statevector()

# Plot the state on the Bloch sphere
plot_bloch_multivector(statevector)
plt.show()

# Create a Quantum Circuit with 2 qubits for entanglement
qc_entangle = QuantumCircuit(2)

# Apply Hadamard gate to the first qubit
qc_entangle.h(0)

# Apply CNOT gate to entangle the qubits
qc_entangle.cx(0, 1)
print("State of entanglement:")
qc_entangle.draw(output='mpl')
plt.show()

# Simulate the entangled circuit
result_entangle = execute(qc_entangle, backend).result()
statevector_entangle = result_entangle.get_statevector()

# Plot the entangled state on the Bloch sphere
plot_bloch_multivector(statevector_entangle)
plt.show()