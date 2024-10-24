import numpy as np

class Qubit:
    def __init__(self, state=0):
        self.state = state  # 0 for |0⟩, 1 for |1⟩

    def __repr__(self):
        return f'|{self.state}⟩'

class SuperpositionQubit:
    def __init__(self, alpha=1/np.sqrt(2), beta=1/np.sqrt(2)):
        # alpha and beta are probability amplitudes
        self.alpha = alpha  # coefficient for |0⟩
        self.beta = beta    # coefficient for |1⟩

    def __repr__(self):
        return f'{self.alpha}|0⟩ + {self.beta}|1⟩'

class EntangledQubits:
    def __init__(self):
        # Create two qubits that are entangled
        self.q1 = SuperpositionQubit()
        self.q2 = SuperpositionQubit()

    def entangle(self):
        # Simple conceptual entanglement: When one qubit is |0⟩, the other is |0⟩, and same for |1⟩
        entangled_state = f'|00⟩ + |11⟩'
        return entangled_state

    def __repr__(self):
        return self.entangle()

# Creating a single qubit
qubit = Qubit(0)
print("Qubit:", qubit)

# Creating a qubit in superposition
superposition_qubit = SuperpositionQubit()
print("Superposition Qubit:", superposition_qubit)

# Creating two entangled qubits
entangled_qubits = EntangledQubits()
print("Entangled Qubits:", entangled_qubits)
