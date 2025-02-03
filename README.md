# Quantum-SWE-Project

This repository contains the code and documentation for the **Quantum Software Engineering** project. The project is undertaken by students **Jeevan Bhatta, Leonardo Ivan Estrella Dzib, Adam Ajroudi,** and **Fernando Araújo**, under the supervision of Professors **Juan Pablo Braña, Alejandra Litterio,** and **Alejandro Fernandez**.

The project explores innovative software engineering approaches—including simulation of networked systems with probabilistic state propagation—that are inspired by quantum computing paradigms and advanced probabilistic models.

## Project Overview
### Quantum

__ To be Added___

### Classical
In this project we develop a simulation framework that models a network of nodes with binary states:
- **0** represents a healthy or functioning node.
- **1** represents a failed node.

The simulation uses an adjacency matrix where each entry (ranging from 0 to 1) encodes the strength of influence that one node has on another. A node’s chance of "failing" (switching from 0 to 1) in the next time step is computed as one minus the product of the probabilities that it remains unaffected by all connected nodes. Once a node fails, it remains failed (an absorbing state).

This approach allows us to study cascaded failures and the interplay between network structure and node behavior, all within a flexible simulation that scales from a small network to systems with *n* nodes.

## Features

- **Input Validation:** Ensures the adjacency matrix is square, values are in [0, 1], and the diagonal is set to 1.
- **Probabilistic Update Mechanism:** Uses a product-of-complements approach to compute the failure probability for each node.
- **Vectorized Operations:** Optimized using NumPy for efficiency and clarity.
- **Reproducible Simulations:** Optional random seed support.
- **Scalability:** Works with arbitrary numbers of nodes (examples provided for 3, 10, and 50 nodes).


## Installation and Usage

1. **Requirements:**  
   - Python 3.x  
   - NumPy

2. **Installation:**  
   Clone the repository and install any required packages (e.g., via pip):

   ```bash
   git clone https://github.com/lifee77/Quantum-SWE-Project.git
   cd Quantum-SWE-Project
   pip install numpy
   ```

3. **Running the Simulation:**  
   Run the simulation script (inside folders classical and quantum):

   ```bash
   python simulation.py
   ```

## Examples for Different Network Sizes

The simulation code has been tested with various network sizes. For instance:

- **n = 3:** Demonstrates the basic functionality.
- **n = 10:** Provides a moderate-sized network simulation.
- **n = 50:** Illustrates scalability and performance with larger networks.

See the examples in the code to explore these scenarios.

## Contributors

- **Students:** Leonardo Ivan Estrella Dzib, Jeevan Bhatta, Adam Ajroudi, Fernando Araújo  
- **Supervisors:** Professors Juan Pablo Braña, Alejandra Litterio, Alejandro Fernandez

Additional contributions and suggestions are welcome. Please feel free to open issues or pull requests.

## Acknowledgements

We thank our supervisors and the entire academic community for their guidance and support during this project.

## License

This project is licensed under the Apache License 2.0.

## Resources

For further details and documentation on quantum software engineering and related simulation techniques, refer to:
- [Quantum Software Engineering Documentation](https://github.com/lifee77/Quantum-SWE-Project)
