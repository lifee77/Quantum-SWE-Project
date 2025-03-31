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
        
        # EXPECTED RESULTS FOR QUANTUM COMPARISON:
        # With the above matrix and initial state [1 0], we expect:
        # - Almost all runs should end in state "11" (both nodes failed)
        # - There should be a small percentage of runs ending in state "10"
        # - The probability of state "11" should be ~99%
    
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
        
        # EXPECTED RESULTS FOR QUANTUM COMPARISON:
        # With the above matrix and initial state [1 0 0 0 0], we expect:
        # - Most runs should end in state "11111" (all nodes failed)
        # - There should be a variety of other states with much smaller percentages
        # - The probability of state "11111" should be ~90-95%
        # - States "11110", "11101", "11011" should also appear with some frequency
    
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
        
        # EXPECTED RESULTS FOR QUANTUM COMPARISON:
        # With the above matrix and initial state [1 0 0 0 0 0 0 0 0 0], we expect:
        # - A wide distribution of final states due to the sparse connections
        # - Highest frequency states should be "1111100100" and "1111101100"
        # - No single state should dominate (>50%), but instead a more even distribution
        # - States with failures in nodes 1, 3, 4, 7 should be most common
    
    return matrix

def main():
    """Run simulations for different node configurations."""
    np.random.seed(42)  # For reproducibility
    
    # Run simulations for 2 nodes
    print("\n" + "=" * 50)
    print("Running simulations for 2 nodes")
    print("=" * 50 + "\n")
    
    n_nodes = 2
    adjacency_matrix = create_adjacency_matrix(n_nodes)
    print(f"Adjacency Matrix for {n_nodes} nodes:")
    print(adjacency_matrix)
    print("\n# REFERENCE FOR QUANTUM COMPARISON:")
    print("# Matrix[0,1] = 0.3 (Node 0 affects Node 1)")
    print("# Matrix[1,0] = 0.7 (Node 1 affects Node 0)")
    print("# Initial State: [1 0] (Node 0 starts failed)")
    
    initial_states = np.zeros(n_nodes, dtype=np.int8)
    initial_states[0] = 1  # First node is failed
    
    state_counts = run_simulation(adjacency_matrix, initial_states)
    create_state_histogram(state_counts, n_nodes)
    
    # Run simulations for 5 nodes
    print("\n" + "=" * 50)
    print("Running simulations for 5 nodes")
    print("=" * 50 + "\n")
    
    n_nodes = 5
    adjacency_matrix = create_adjacency_matrix(n_nodes)
    print(f"Adjacency Matrix for {n_nodes} nodes:")
    print(adjacency_matrix)
    print("\n# REFERENCE FOR QUANTUM COMPARISON:")
    print("# Matrix[1,0] = 0.8, Matrix[3,0] = 0.4 (Node 0 affects Nodes 1,3)")
    print("# Matrix[2,1] = 0.6, Matrix[4,1] = 0.4 (Node 1 affects Nodes 2,4)")
    print("# Matrix[3,2] = 0.7, Matrix[4,2] = 0.2 (Node 2 affects Nodes 3,4)")
    print("# Matrix[0,3] = 0.3, Matrix[2,3] = 0.2, Matrix[4,3] = 0.5 (Node 3 affects Nodes 0,2,4)")
    print("# Matrix[1,4] = 0.5, Matrix[2,4] = 0.1 (Node 4 affects Nodes 1,2)")
    print("# Initial State: [1 0 0 0 0] (Node 0 starts failed)")
    
    initial_states = np.zeros(n_nodes, dtype=np.int8)
    initial_states[0] = 1  # First node is failed
    
    state_counts = run_simulation(adjacency_matrix, initial_states)
    create_state_histogram(state_counts, n_nodes)
    
    # Run simulations for 10 nodes
    print("\n" + "=" * 50)
    print("Running simulations for 10 nodes")
    print("=" * 50 + "\n")
    
    n_nodes = 10
    adjacency_matrix = create_adjacency_matrix(n_nodes)
    print(f"Adjacency Matrix for {n_nodes} nodes:")
    print(adjacency_matrix)
    print("\n# REFERENCE FOR QUANTUM COMPARISON:")
    print("# Matrix[1,0] = 0.7, Matrix[3,0] = 0.4, Matrix[7,0] = 0.2 (Node 0 affects Nodes 1,3,7)")
    print("# Matrix[2,1] = 0.5 (Node 1 affects Node 2)")
    print("# Matrix[6,2] = 0.3, Matrix[8,2] = 0.3 (Node 2 affects Nodes 6,8)")
    print("# Matrix[4,3] = 0.8 (Node 3 affects Node 4)")
    print("# Matrix[7,4] = 0.5 (Node 4 affects Node 7)")
    print("# Matrix[1,5] = 0.6, Matrix[9,5] = 0.6 (Node 5 affects Nodes 1,9)")
    print("# Matrix[5,8] = 0.4 (Node 8 affects Node 5)")
    print("# Matrix[6,9] = 0.7 (Node 9 affects Node 6)")
    print("# Initial State: [1 0 0 0 0 0 0 0 0 0] (Node 0 starts failed)")
    
    initial_states = np.zeros(n_nodes, dtype=np.int8)
    initial_states[0] = 1  # First node is failed
    
    state_counts = run_simulation(adjacency_matrix, initial_states)
    create_state_histogram(state_counts, n_nodes)
    
    # Save the adjacency matrices to files for quantum side reference
    for n in [2, 5, 10]:
        adj_matrix = create_adjacency_matrix(n)
        output_dir = "classical/results/"
        os.makedirs(output_dir, exist_ok=True)
        np.savetxt(f"{output_dir}adjacency_matrix_n{n}.txt", adj_matrix, fmt="%.2f")
        
    print("\n" + "=" * 50)
    print("REFERENCE FOR QUANTUM COMPARISON:")
    print("=" * 50)
    print("The adjacency matrices have been saved to the classical/results/ directory")
    print("for use by the quantum simulation code. Use these files to ensure")
    print("that the quantum simulations use the exact same probabilities as")
    print("the classical simulations for direct comparison.")
    print("=" * 50) 