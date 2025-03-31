# REFERENCE DATA FOR QUANTUM COMPARISON

This file contains reference data from the classical simulations to use for
direct comparison with quantum simulation results.

## ADJACENCY MATRICES

### 2-NODE SYSTEM:
```
Matrix = [
    [1.0, 0.3],
    [0.7, 1.0]
]
```

Initial State: `[1, 0]` (Node 0 starts failed)

#### Interpretation:
- Node 0 affects Node 1 with 30% probability (Matrix[0,1] = 0.3)
- Node 1 affects Node 0 with 70% probability (Matrix[1,0] = 0.7)
- Diagonal elements (1.0) indicate that a failed node stays failed

#### Expected Results:
- Almost all runs end in state "11" (both nodes failed)
- A small percentage of runs end in state "10"
- The probability of state "11" is approximately 99%


### 5-NODE SYSTEM:
```
Matrix = [
    [1.0, 0.0, 0.0, 0.3, 0.0],
    [0.8, 1.0, 0.0, 0.0, 0.5],
    [0.0, 0.6, 1.0, 0.2, 0.1],
    [0.4, 0.0, 0.7, 1.0, 0.0],
    [0.0, 0.4, 0.2, 0.5, 1.0]
]
```

Initial State: `[1, 0, 0, 0, 0]` (Node 0 starts failed)

#### Interpretation:
- Node 0 affects Nodes 1 and 3 (Matrix[1,0] = 0.8, Matrix[3,0] = 0.4)
- Node 1 affects Nodes 2 and 4 (Matrix[2,1] = 0.6, Matrix[4,1] = 0.4)
- Node 2 affects Nodes 3 and 4 (Matrix[3,2] = 0.7, Matrix[4,2] = 0.2)
- Node 3 affects Nodes 0, 2, and 4 (Matrix[0,3] = 0.3, Matrix[2,3] = 0.2, Matrix[4,3] = 0.5)
- Node 4 affects Nodes 1 and 2 (Matrix[1,4] = 0.5, Matrix[2,4] = 0.1)

#### Expected Results:
- Most runs end in state "11111" (all nodes failed) ~90-95%
- States like "11110", "11101", and "11011" also appear with some frequency
- Distribution shows the cascading nature of failures through the network


### 10-NODE SYSTEM:
```
Matrix = [
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.7, 1.0, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.4, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.4, 0.0],
    [0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.7],
    [0.2, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 1.0]
]
```

Initial State: `[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]` (Node 0 starts failed)

#### Interpretation:
- Node 0 affects Nodes 1, 3, 7 (Matrix[1,0] = 0.7, Matrix[3,0] = 0.4, Matrix[7,0] = 0.2)
- Node 1 affects Node 2 (Matrix[2,1] = 0.5)
- Node 2 affects Nodes 6, 8 (Matrix[6,2] = 0.3, Matrix[8,2] = 0.3)
- Node 3 affects Node 4 (Matrix[4,3] = 0.8)
- Node 4 affects Node 7 (Matrix[7,4] = 0.5)
- Node 5 affects Nodes 1, 9 (Matrix[1,5] = 0.6, Matrix[9,5] = 0.6)
- Node 8 affects Node 5 (Matrix[5,8] = 0.4)
- Node 9 affects Node 6 (Matrix[6,9] = 0.7)

#### Expected Results:
- A wide distribution of final states due to the sparse connections
- Most common states are "1111100100" and "1111101100" (~10-20% each)
- No single state dominates the distribution
- States with failures in nodes 1, 3, 4, 7 are most common


## COMPARISON INSTRUCTIONS

1. Use the exact same adjacency matrices defined above in your quantum simulations
2. Initialize with the same initial states as the classical simulations
3. Run the quantum simulations for 1000 shots
4. Compare the state distributions between classical and quantum results
5. Create histograms with the same formatting for direct visual comparison
6. Key differences to observe:
   - Are the quantum simulation probabilities consistent with classical results?
   - Are there quantum-specific patterns in the distribution?
   - Does the quantum simulation converge faster or show different behavior?

> **NOTE**: The files `classical_simulation_n2.png`, `classical_simulation_n5.png`, and
`classical_simulation_n10.png` contain the histogram plots from the classical simulations. 