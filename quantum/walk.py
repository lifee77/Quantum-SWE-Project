from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import UnitaryGate
import numpy as np

##############################################################################
# 1. Define the classical data and normalize the transition (adjacency) matrix
##############################################################################

# Your original weighted adjacency matrix (risk propagation weights)
adjacency_matrix = np.array([[0,    0.10, 0.13],
                             [0.20, 0.0,  0.50],
                             [0.17, 0.30, 0.0]])

# For each vertex, we will use its (nonzero) row to define transition probabilities.
# Normalize each row (if nonzero) so that the two outgoing weights sum to 1.
P = np.zeros_like(adjacency_matrix)
for i in range(adjacency_matrix.shape[0]):
    row_sum = np.sum(adjacency_matrix[i])
    if row_sum > 0:
        P[i] = adjacency_matrix[i] / row_sum
    else:
        P[i] = adjacency_matrix[i]

# For our 3–vertex graph we fix the neighbor mapping as follows:
#   Vertex 0: coin outcome 0  --> goes to vertex 1, and coin outcome 1  --> goes to vertex 2.
#   Vertex 1: coin outcome 0  --> goes to vertex 0, and coin outcome 1  --> goes to vertex 2.
#   Vertex 2: coin outcome 0  --> goes to vertex 0, and coin outcome 1  --> goes to vertex 1.

##############################################################################
# 2. Define the coin operators for each vertex
##############################################################################

def coin_operator(i):
    """
    For vertex i, returns a 2x2 unitary that sends |0> to
      sqrt(p0) |0> + sqrt(p1) |1>,
    where p0 and p1 are the normalized probabilities for the two outgoing edges.
    """
    if i == 0:
        p0 = P[0,1]  # probability to go to vertex 1
        p1 = P[0,2]  # probability to go to vertex 2
    elif i == 1:
        p0 = P[1,0]  # probability to go to vertex 0
        p1 = P[1,2]  # probability to go to vertex 2
    elif i == 2:
        p0 = P[2,0]  # probability to go to vertex 0
        p1 = P[2,1]  # probability to go to vertex 1
    # We want a rotation that takes |0> to:
    #    cos(theta/2)|0> + sin(theta/2)|1>,
    # so that cos^2(theta/2)=p0 and sin^2(theta/2)=p1.
    theta = 2 * np.arcsin(np.sqrt(p1))
    # Return the standard rotation matrix R_y(theta)
    return np.array([[np.cos(theta/2), -np.sin(theta/2)],
                     [np.sin(theta/2),  np.cos(theta/2)]], dtype=complex)

# Build the block-diagonal coin operator on the 6-dimensional edge space.
# The edge space is spanned by:
#   |0, coin=0> : vertex 0, coin=0   (edge (0→1))
#   |0, coin=1> : vertex 0, coin=1   (edge (0→2))
#   |1, coin=0> : vertex 1, coin=0   (edge (1→0))
#   |1, coin=1> : vertex 1, coin=1   (edge (1→2))
#   |2, coin=0> : vertex 2, coin=0   (edge (2→0))
#   |2, coin=1> : vertex 2, coin=1   (edge (2→1))
C = np.zeros((6, 6), dtype=complex)
for i in range(3):
    Ci = coin_operator(i)  # 2x2 for vertex i
    if i == 0:
        C[0:2, 0:2] = Ci
    elif i == 1:
        C[2:4, 2:4] = Ci
    elif i == 2:
        C[4:6, 4:6] = Ci

##############################################################################
# 3. Define the shift (flip–flop) operator S on the edge space.
#    It “swaps” the edge (i, j) to (j, i). In our encoding the mapping is:
#
#    S: |0, coin=0> (edge (0→1))  -->  |1, coin=0> (edge (1→0))
#       |0, coin=1> (edge (0→2))  -->  |2, coin=0> (edge (2→0))
#       |1, coin=0> (edge (1→0))  -->  |0, coin=0> (edge (0→1))
#       |1, coin=1> (edge (1→2))  -->  |2, coin=1> (edge (2→1))
#       |2, coin=0> (edge (2→0))  -->  |0, coin=1> (edge (0→2))
#       |2, coin=1> (edge (2→1))  -->  |1, coin=1> (edge (1→2))
##############################################################################
S = np.zeros((6,6), dtype=complex)
S[0,2] = 1.0  # maps state 0 -> state 2
S[1,4] = 1.0  # maps state 1 -> state 4
S[2,0] = 1.0  # maps state 2 -> state 0
S[3,5] = 1.0  # maps state 3 -> state 5
S[4,1] = 1.0  # maps state 4 -> state 1
S[5,3] = 1.0  # maps state 5 -> state 3

##############################################################################
# 4. Combine coin and shift to define one step of the quantum walk:
#       U_qw = S * C
##############################################################################
U_qw = S @ C  # (6x6 unitary on the edge space)

##############################################################################
# 5. Embed the 6-dimensional unitary into an 8-dimensional (3-qubit) unitary.
#    (Here we simply set the action on the two extra basis states to be the identity.)
##############################################################################
U_embed = np.eye(8, dtype=complex)
U_embed[:6, :6] = U_qw

##############################################################################
# 6. Build the Qiskit circuit
##############################################################################

# We will use 3 qubits to encode the 8-dimensional Hilbert space.
qr = QuantumRegister(3, 'qw')
cr = ClassicalRegister(3, 'c')
qc = QuantumCircuit(qr, cr)

# --- Initial state preparation ---
# We want to “encode” the initial risk vector into the vertices.
# In the flip–flop walk the state lives on edges. One common choice is to
# start with, for each vertex i, an equal superposition over the two coin states.
# Thus, for vertex i with risk r_i, we set the amplitude on both edges out of i to:
#    sqrt(r_i)/sqrt(2)
risk_vector = np.array([0.2, 0.3, 0.5])
initial_state = np.zeros(8, dtype=complex)
# Our encoding (edge ordering) is:
#   state 0: vertex 0, coin=0
#   state 1: vertex 0, coin=1
#   state 2: vertex 1, coin=0
#   state 3: vertex 1, coin=1
#   state 4: vertex 2, coin=0
#   state 5: vertex 2, coin=1
initial_state[0] = np.sqrt(risk_vector[0]) / np.sqrt(2)
initial_state[1] = np.sqrt(risk_vector[0]) / np.sqrt(2)
initial_state[2] = np.sqrt(risk_vector[1]) / np.sqrt(2)
initial_state[3] = np.sqrt(risk_vector[1]) / np.sqrt(2)
initial_state[4] = np.sqrt(risk_vector[2]) / np.sqrt(2)
initial_state[5] = np.sqrt(risk_vector[2]) / np.sqrt(2)
# States 6 and 7 remain 0 (unused)

# Use Qiskit's initialize function to set the state.
qc.initialize(initial_state, qr)

# --- One quantum walk step ---
# We now apply the unitary gate representing one step of the flip–flop quantum walk.
# Here we create the unitary gate using the updated UnitaryGate import.
qw_gate = UnitaryGate(U_embed, label="QW_Step")
qc.append(qw_gate, qr)

# --- Measurement ---
qc.measure(qr, cr)

# Draw the circuit
qc.draw("mpl")
