#!/usr/bin/env python3

"""
Classical Simulation for Attack Propagation - 10 Node 1-Step
This module simulates the 1-step propagation of failures in a classical system with 10 nodes.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

def create_state_histogram(state_counts, n_nodes, output_dir="classical/results/"):
    """
    Create a histogram plot of state occurrences.
    
    Args:
        state_counts (dict): Dictionary of states and their counts
        n_nodes (int): Number of nodes in the system
        output_dir (str): Directory to save the plot
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort states for better visualization
    sorted_states = sorted(state_counts.items(), key=lambda x: int(x[0], 2))
    states, counts = zip(*sorted_states) if sorted_states else ([], [])
    
    # Calculate probabilities for each state
    total_runs = sum(counts)
    probabilities = [count / total_runs for count in counts]
    
    # Create figure and axes
    plt.figure(figsize=(12, 6))
    
    # Plot histogram with blue bars
    bars = plt.bar(range(len(states)), probabilities, color='royalblue', alpha=0.7)
    
    # Add text annotations inside bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height/2,
                 f'{count}', ha='center', va='center', color='white', fontweight='bold')
    
    # Set labels and title
    plt.title(f'State Distribution for Classical System with {n_nodes} Nodes (1-Step)')
    plt.xlabel('State')
    plt.ylabel('Probability')
    
    # Set x-tick labels to show binary states
    plt.xticks(range(len(states)), states, rotation=45)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f"{output_dir}classical_simulation_n{n_nodes}_1step.png")
    plt.close()

def create_adjacency_matrix(n=10):
    """
    Create a realistic adjacency matrix for 10 nodes.
    
    Returns:
        np.ndarray: Adjacency matrix with probabilities
    """
    # Create identity matrix (diagonal elements are 1)
    matrix = np.eye(n)
    
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

def compute_update_probability(states, adjacency_matrix):
    """
    Compute the update probability for each node based on current states.
    
    Args:
        states (np.ndarray): Current states of all nodes (binary: 0 or 1)
        adjacency_matrix (np.ndarray): Adjacency matrix of probabilities
        
    Returns:
        np.ndarray: Update probabilities for each node
    """
    # For each node i, compute the product over all j of (1 - a_ij * state_j)
    non_failure_prob = np.prod(1 - adjacency_matrix * states, axis=1)
    return 1 - non_failure_prob

def run_simulation(adjacency_matrix, initial_states, num_iterations=1000, num_time_steps=1):
    """
    Run multiple iterations of the simulation and collect statistics
    
    Args:
        adjacency_matrix (np.ndarray): The adjacency matrix
        initial_states (np.ndarray): Initial states of nodes
        num_iterations (int): Number of simulation iterations
        num_time_steps (int): Number of time steps per iteration
        
    Returns:
        dict: Dictionary of final states and their counts
    """
    n_nodes = len(initial_states)
    state_counts = {}
    run_histories = []
    
    print(f"Initial States:\n{initial_states}")
    
    # Calculate and print initial update probabilities
    update_probs = compute_update_probability(initial_states, adjacency_matrix)
    print("\nInitial Update Probabilities:")
    print(f"Node States: {initial_states}, Update Probabilities: {update_probs}")
    
    for i in range(num_iterations):
        if i % 100 == 0:
            print(f"Completed {i}/{num_iterations} simulations for n={n_nodes} nodes")
        
        # Initialize state for this run
        states = initial_states.copy()
        history = [states.copy()]
        
        # Run simulation for specified time steps (1 step for this script)
        for _ in range(num_time_steps):
            # Compute update probabilities
            update_probs = compute_update_probability(states, adjacency_matrix)
            
            # Generate random values and update states
            random_vals = np.random.rand(n_nodes)
            new_states = np.where(states == 1, 1, (random_vals < update_probs).astype(np.int8))
            states = new_states
            history.append(states.copy())
        
        # Convert final state to binary string for counting
        final_state = ''.join(map(str, states))
        state_counts[final_state] = state_counts.get(final_state, 0) + 1
        
        # Store run history for first few runs
        if i < 3:
            run_histories.append(history)
    
    print(f"\nFinal State Distribution:")
    for state, count in sorted(state_counts.items(), key=lambda x: int(x[0], 2)):
        percentage = (count / num_iterations) * 100
        print(f"State {state}: {count} occurrences ({percentage:.1f}%)")
    
    print("\nExample Run Histories (first 3):")
    for i, history in enumerate(run_histories):
        print(f"Run {i + 1}:")
        for t, state in enumerate(history):
            print(f"  Time step {t}: {state}")
        print()
    
    return state_counts

def main():
    """Run simulation for 10 nodes system with 1 time step."""
    np.random.seed(42)  # For reproducibility
    
    # Run simulations for 10 nodes
    print("\n" + "=" * 50)
    print("Running simulation for 10 nodes (1 time step)")
    print("=" * 50 + "\n")
    
    n_nodes = 10
    adjacency_matrix = create_adjacency_matrix(n_nodes)
    print(f"Adjacency Matrix for {n_nodes} nodes:")
    print(adjacency_matrix)
    print("\n# REFERENCE FOR QUANTUM COMPARISON (1 TIME STEP):")
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
    
    print("\n" + "=" * 50)
    print("REFERENCE FOR QUANTUM COMPARISON (1 TIME STEP):")
    print("=" * 50)
    print("The 1-step simulation demonstrates the direct effect of probabilities.")
    print("With node 0 initially failed, we expect approximately:")
    print("- 70% of runs to show node 1 also failed")
    print("- 40% of runs to show node 3 also failed")
    print("- 20% of runs to show node 7 also failed")
    print("This distribution directly reflects the values in the adjacency matrix.")
    print("=" * 50)

if __name__ == "__main__":
    main() 