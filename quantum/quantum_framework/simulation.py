# simulation.py
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer
from qiskit.circuit.library import UnitaryGate
from dilation import insert_non_unitary

def build_quantum_circuit(num_qubits, non_unitary_operator, states, add_measurements=True):
    """
    Builds a quantum circuit that applies the dilated non-unitary operator.
    
    Parameters:
        num_qubits (int): Number of main qubits.
        non_unitary_operator (np.ndarray): The operator (2^n x 2^n) before dilation.
        states (list): List of state vectors for initializing each main qubit.
        add_measurements (bool): Whether to add measurement gates.
    
    Returns:
        qc (QuantumCircuit): The constructed quantum circuit.
    """
    main = QuantumRegister(num_qubits, 'main')
    ancilla = QuantumRegister(1, 'ancilla')
    main_cl = ClassicalRegister(num_qubits, 'main_cl')
    ancilla_cl = ClassicalRegister(1, 'ancilla_cl')
    
    qc = QuantumCircuit(main, ancilla, main_cl, ancilla_cl)
    
    # Initialize main qubits
    for i, state in enumerate(states):
        qc.initialize(state, main[i])
    
    # Embed non-unitary operator using dilation
    UA = insert_non_unitary(non_unitary_operator)
    gate = UnitaryGate(UA)
    qc.append(gate, list(range(num_qubits + 1)))
    
    if add_measurements:
        qc.measure(main, main_cl)
        qc.measure(ancilla, ancilla_cl)
    
    return qc

def run_simulation(qc, shots=1024):
    """
    Runs a QASM simulation using Qiskit's Aer simulator.
    
    Parameters:
        qc (QuantumCircuit): The circuit to simulate.
        shots (int): Number of shots.
    
    Returns:
        counts (dict): Dictionary of measurement counts.
    """
    simulator = Aer.get_backend('aer_simulator')
    compiled = transpile(qc, simulator)
    result = simulator.run(compiled, shots=shots).result()
    return result.get_counts()

def run_statevector_simulation(qc):
    """
    Runs a statevector simulation (measurements removed).
    
    Returns:
        statevector (dict): Dictionary of probabilities for each basis state.
    """
    from qiskit.quantum_info import Statevector
    qc_no_meas = qc.remove_final_measurements(inplace=False)
    state = Statevector.from_instruction(qc_no_meas)
    return state.probabilities_dict()
