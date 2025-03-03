import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import Aer
import matplotlib.pyplot as plt

def shift_plus(N):
    """Return the N×N shift-plus matrix: |i> -> |(i+1) mod N>."""
    S = np.zeros((N, N))
    for i in range(N):
        S[i, (i+1)%N] = 1
    return S

def shift_minus(N):
    """Return the N×N shift-minus matrix: |i> -> |(i-1) mod N>."""
    S = np.zeros((N, N))
    for i in range(N):
        S[i, (i-1)%N] = 1
    return S

def build_shift_operator(num_position_qubits):
    """
    Build the composite shift operator for a quantum walk on a cycle.
    
    Uses one coin qubit and a position register with dimension N = 2^(num_position_qubits).
    Returns an 8×8 unitary if num_position_qubits = 2.
    """
    N = 2**num_position_qubits
    S_plus = shift_plus(N)
    S_minus = shift_minus(N)
    # Coin projectors
    P0 = np.array([[1, 0], [0, 0]])  # acts when coin is |0>
    P1 = np.array([[0, 0], [0, 1]])  # acts when coin is |1>
    # Composite shift operator:
    S = np.kron(P0, S_plus) + np.kron(P1, S_minus)
    return S

def build_quantum_walk_circuit(num_steps, num_position_qubits=2):
    """
    Build a discrete-time quantum walk circuit on a cycle.
    
    The circuit uses:
    - 1 coin qubit.
    - A position register with num_position_qubits (N = 2^(num_position_qubits) positions).
    
    Each step applies a Hadamard gate to the coin followed by a shift operator.
    """
    # Registers: coin (1 qubit), position (num_position_qubits qubits)
    coin = QuantumRegister(1, 'coin')
    position = QuantumRegister(num_position_qubits, 'pos')
    coin_cl = ClassicalRegister(1, 'coin_cl')
    pos_cl = ClassicalRegister(num_position_qubits, 'pos_cl')
    
    qc = QuantumCircuit(coin, position, coin_cl, pos_cl)
    
    # Optional: Initialize coin in superposition
    qc.h(coin)
    
    for step in range(num_steps):
        # Apply coin toss (Hadamard on coin)
        qc.h(coin)
        
        # Build and apply the shift operator (controlled by the coin)
        S = build_shift_operator(num_position_qubits)
        S_gate = UnitaryGate(S, label=f"Shift_{step}")
        # Append S_gate on coin + position (note the register order: coin then pos)
        qc.append(S_gate, coin[:] + position[:])
    
    # Measure coin and position registers
    qc.measure(coin, coin_cl)
    qc.measure(position, pos_cl)
    
    return qc

def run_quantum_walk(qc, shots=1024):
    """Simulate the quantum walk circuit using Qiskit's Aer simulator."""
    simulator = Aer.get_backend('aer_simulator')
    compiled = transpile(qc, simulator)
    result = simulator.run(compiled, shots=shots).result()
    return result.get_counts()

if __name__ == "__main__":
    num_steps = 5
    num_position_qubits = 2  # For a cycle with 4 positions
    qc_walk = build_quantum_walk_circuit(num_steps, num_position_qubits)
    qc_walk.draw('mpl')  # This displays the circuit diagram
    
    counts = run_quantum_walk(qc_walk, shots=1024)
    print("Quantum walk counts:", counts)
    
    # Optionally, plot a histogram
    from qiskit.visualization import plot_histogram
    plot_histogram(counts)
    plt.show()
