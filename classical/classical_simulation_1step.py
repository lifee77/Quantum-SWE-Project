import numpy as np

def create_adjacency_matrix(n_nodes):
    # This function is not provided in the original file or the code block
    # It's assumed to exist as it's called in the main function
    pass

def run_simulation(adjacency_matrix, initial_states):
    # This function is not provided in the original file or the code block
    # It's assumed to exist as it's called in the main function
    pass

def create_state_histogram(state_counts, n_nodes):
    # This function is not provided in the original file or the code block
    # It's assumed to exist as it's called in the main function
    pass

def main():
    """Run simulations for different node configurations with 1 time step."""
    np.random.seed(42)  # For reproducibility
    
    # Run simulations for 2 nodes
    print("\n" + "=" * 50)
    print("Running simulations for 2 nodes (1 time step)")
    print("=" * 50 + "\n")
    
    n_nodes = 2
    adjacency_matrix = create_adjacency_matrix(n_nodes)
    print(f"Adjacency Matrix for {n_nodes} nodes:")
    print(adjacency_matrix)
    print("\n# REFERENCE FOR QUANTUM COMPARISON (1 TIME STEP):")
    print("# Matrix[0,1] = 0.3 (Node 0 affects Node 1)")
    print("# Matrix[1,0] = 0.7 (Node 1 affects Node 0)")
    print("# Initial State: [1 0] (Node 0 starts failed)")
    
    initial_states = np.zeros(n_nodes, dtype=np.int8)
    initial_states[0] = 1  # First node is failed
    
    state_counts = run_simulation(adjacency_matrix, initial_states)
    create_state_histogram(state_counts, n_nodes)
    
    # Run simulations for 5 nodes
    print("\n" + "=" * 50)
    print("Running simulations for 5 nodes (1 time step)")
    print("=" * 50 + "\n")
    
    n_nodes = 5
    adjacency_matrix = create_adjacency_matrix(n_nodes)
    print(f"Adjacency Matrix for {n_nodes} nodes:")
    print(adjacency_matrix)
    
    initial_states = np.zeros(n_nodes, dtype=np.int8)
    initial_states[0] = 1  # First node is failed
    
    state_counts = run_simulation(adjacency_matrix, initial_states)
    create_state_histogram(state_counts, n_nodes)
    
    # Run simulations for 10 nodes
    print("\n" + "=" * 50)
    print("Running simulations for 10 nodes (1 time step)")
    print("=" * 50 + "\n")
    
    n_nodes = 10
    adjacency_matrix = create_adjacency_matrix(n_nodes)
    print(f"Adjacency Matrix for {n_nodes} nodes:")
    print(adjacency_matrix)
    
    initial_states = np.zeros(n_nodes, dtype=np.int8)
    initial_states[0] = 1  # First node is failed
    
    state_counts = run_simulation(adjacency_matrix, initial_states)
    create_state_histogram(state_counts, n_nodes)
    
    print("\n" + "=" * 50)
    print("REFERENCE FOR QUANTUM COMPARISON (1 TIME STEP):")
    print("=" * 50)
    print("The 1-step simulation demonstrates the direct effect of the probabilities")
    print("in the adjacency matrix without cumulative effects over multiple steps.")
    print("For 2 nodes with Matrix[0,1]=0.3, we expect 30% of runs to end in state '11'")
    print("and 70% to remain in state '10', directly showing the probability value.")
    print("=" * 50)

if __name__ == "__main__":
    main() 