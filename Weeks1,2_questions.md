**Qubit:** [Further Details](https://www.ibm.com/topics/qubit) <br>
A qubit, or quantum bit, is the basic unit of information used to encode data in quantum computing and can be best understood as the quantum equivalent of the traditional bit used by classical computers to encode information in binary.
<br>Qubits are generally, although not exclusively, created by manipulating and measuring quantum particles (the smallest known building blocks of the physical universe), such as photons, electrons, trapped ions, superconducting circuits, and atoms. 

*How is a Qbit different from bit?*<br>
While each bit can be either a 0 or a 1, a single qubit can be either a 0, a 1, or a superposition.<br>
* It is not possible to determine if a qbit is in the state 0 or 1 unlike in a bit. We would rather need to find the values of alpha nd beta.
*Even when we measure a qbit's state, it gives us 0 or 1 with  aprobability value attached to it.

<br>
A quantum superposition can be described as *both 0 and 1, or as all the possible states between 0 and 1* because it actually represents the probability of the qubit’s state. 
![Notation for bits](image-1.png)
**Quantum Superposition:** <br>
It is the state of a qubit formed by the linear combination of the two regular states<br>
alpha * 0 + beta*1 <br>
The two states Cat 0 and Cat 1 form orthonormal (orthogonal to each other with a magnitude of 1 as seen in the image above) bases for the vector space (of Superposition).

1. **Hadamard Gate**:
   - The Hadamard gate (H gate) is a single-qubit gate that creates a superposition state from a basis state. It transforms the basis states |0⟩ and |1⟩ into an equal superposition of both states.
   - Mathematically, it is represented as:
     \[
     H = \frac{1}{\sqrt{2}} \begin{pmatrix}
     1 & 1 \\
     1 & -1
     \end{pmatrix}
     \]

2. **State of Superposition**:
   - Superposition is a fundamental principle of quantum mechanics where a quantum system can be in multiple states at once. For a qubit, this means it can be in a combination of the |0⟩ and |1⟩ states.
   - A general superposition state is written as:
     \[
     |\psi⟩ = \alpha|0⟩ + \beta|1⟩
     \]
     where \(\alpha\) and \(\beta\) are complex numbers such that \(|\alpha|^2 + |\beta|^2 = 1\).

3. **State of Entanglement**:
   - Entanglement is a quantum phenomenon where the states of two or more qubits become interconnected such that the state of one qubit cannot be described independently of the state of the other qubits.
   - An example of an entangled state is the Bell state:
     \[
     |\Phi^+\rangle = \frac{1}{\sqrt{2}} (|00\rangle + |11\rangle)
     \]

4. **Other Quantum Gates for Basic Tasks**:
   - **Pauli-X Gate (NOT Gate)**: Flips the state of a qubit (|0⟩ to |1⟩ and |1⟩ to |0⟩).
     \[
     X = \begin{pmatrix}
     0 & 1 \\
     1 & 0
     \end{pmatrix}
     \]
   - **Pauli-Y Gate**: Similar to the X gate but introduces a phase shift.
     \[
     Y = \begin{pmatrix}
     0 & -i \\
     i & 0
     \end{pmatrix}
     \]
   - **Pauli-Z Gate**: Introduces a phase shift to the |1⟩ state.
     \[
     Z = \begin{pmatrix}
     1 & 0 \\
     0 & -1
     \end{pmatrix}
     \]
   - **CNOT Gate (Controlled-NOT Gate)**: A two-qubit gate that flips the second qubit (target) if the first qubit (control) is |1⟩.
     \[
     \text{CNOT} = \begin{pmatrix}
     1 & 0 & 0 & 0 \\
     0 & 1 & 0 & 0 \\
     0 & 0 & 0 & 1 \\
     0 & 0 & 1 & 0
     \end{pmatrix}
     \]
   - **Phase Gate**: Introduces a phase shift to the qubit.
     \[
     S = \begin{pmatrix}
     1 & 0 \\
     0 & i
     \end{pmatrix}
     \]
