#!/usr/bin/env python3

"""
Export Adjacency Matrices for Quantum Comparison
This script exports the adjacency matrices used in classical simulations 
as NumPy files that can be easily loaded by the quantum simulations.
"""

import os
import numpy as np

def create_adjacency_matrix(n):
    """
    Create a realistic adjacency matrix for n nodes.
    
    Args:
        n (int): Number of nodes
        
    Returns:
        np.ndarray: Adjacency matrix with probabilities
    """
    # Create identity matrix (diagonal elements are 1)
    matrix = np.eye(n)
    
    if n == 2:
        # For 2 nodes, each affects the other with different probabilities
        matrix[0, 1] = 0.3  # Node 0 affects Node 1 with 30% probability
        matrix[1, 0] = 0.7  # Node 1 affects Node 0 with 70% probability
    
    elif n == 5:
        # For 5 nodes, create a more complex network
        # Node 0 affects nodes 1 and 3
        matrix[1, 0] = 0.8
        matrix[3, 0] = 0.4
        
        # Node 1 affects nodes 2 and 4
        matrix[2, 1] = 0.6
        matrix[4, 1] = 0.4
        
        # Node 2 affects nodes 3 and 4
        matrix[3, 2] = 0.7
        matrix[4, 2] = 0.2
        
        # Node 3 affects nodes 0, 2 and 4
        matrix[0, 3] = 0.3
        matrix[2, 3] = 0.2
        matrix[4, 3] = 0.5
        
        # Node 4 affects nodes 1, 2
        matrix[1, 4] = 0.5
        matrix[2, 4] = 0.1
    
    elif n == 10:
        # For 10 nodes, establish a sparse network where only specific nodes affect others
        # Node 0 affects nodes 1, 3, 7
        matrix[1, 0] = 0.7
        matrix[3, 0] = 0.4
        matrix[7, 0] = 0.2
        
        # Node 1 affects node 2
        matrix[2, 1] = 0.5
        
        # Node 2 affects nodes 6, 8
        matrix[6, 2] = 0.3
        matrix[8, 2] = 0.3
        
        # Node 3 affects node 4
        matrix[4, 3] = 0.8
        
        # Node 4 affects node 7
        matrix[7, 4] = 0.5
        
        # Node 5 affects nodes 1, 9
        matrix[1, 5] = 0.6
        matrix[9, 5] = 0.6
        
        # Node 8 affects node 5
        matrix[5, 8] = 0.4
        
        # Node 9 affects node 6
        matrix[6, 9] = 0.7
    
    return matrix

def create_initial_state(n):
    """
    Create initial state vector with only the first node failed.
    
    Args:
        n (int): Number of nodes
        
    Returns:
        np.ndarray: Initial state vector
    """
    state = np.zeros(n, dtype=np.int8)
    state[0] = 1  # First node is failed
    return state

def main():
    """Export adjacency matrices and initial states for all node configurations."""
    output_dir = "classical/results/"
    os.makedirs(output_dir, exist_ok=True)
    
    # Export adjacency matrices and initial states for 2, 5, and 10 nodes
    for n in [2, 5, 10]:
        # Create and save adjacency matrix
        adj_matrix = create_adjacency_matrix(n)
        np.save(f"{output_dir}adjacency_matrix_n{n}.npy", adj_matrix)
        
        # Save as text file for human readability
        np.savetxt(f"{output_dir}adjacency_matrix_n{n}.txt", adj_matrix, fmt="%.2f")
        
        # Create and save initial state
        initial_state = create_initial_state(n)
        np.save(f"{output_dir}initial_state_n{n}.npy", initial_state)
        
        print(f"Exported adjacency matrix and initial state for {n} nodes")
        print(f"- Matrix shape: {adj_matrix.shape}")
        print(f"- Initial state: {initial_state}")
    
    print("\nAll files exported successfully to:", output_dir)
    print("Files can be loaded in Python with: np.load('path/to/file.npy')")

if __name__ == "__main__":
    main() 