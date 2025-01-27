import numpy as np
import random

def initialize_system(n):
    """
    Initializes the system with n nodes, all starting in state 0 (functioning).

    Args:
        n: The number of nodes in the system.

    Returns:
        A tuple containing:
        - states: A NumPy array representing the initial states of the nodes.
        - adjacency_matrix: A randomly generated n x n adjacency matrix.
    """
    states = np.zeros(n)  # All nodes start in state 0
    adjacency_matrix = np.random.rand(n, n)  # Random probabilities between 0 and 1
    np.fill_diagonal(adjacency_matrix, 1)  # Ensure a_ii = 1
    return states, adjacency_matrix

def update_node_state(i, states, adjacency_matrix):
    """
    Calculates the probability of node i failing at the next time step and updates its state.

    Args:
        i: The index of the node to update.
        states: The current states of all nodes.
        adjacency_matrix: The adjacency matrix.

    Returns:
        The updated state of node i (0 or 1).
    """
    if states[i] == 1:
        return 1  # Node remains failed if it was already failed

    product = 1
    for j in range(len(states)):
        product *= (1 - adjacency_matrix[i, j] * states[j])

    probability_of_failure = 1 - product

    # Flip a coin with the calculated probability to determine the new state
    if random.random() < probability_of_failure:
        return 1
    else:
        return 0

def simulate_time_step(states, adjacency_matrix):
    """
    Simulates one time step for the entire system.

    Args:
        states: The current states of all nodes.
        adjacency_matrix: The adjacency matrix.

    Returns:
        A NumPy array representing the updated states of the nodes after one time step.
    """
    new_states = np.zeros_like(states)
    for i in range(len(states)):
        new_states[i] = update_node_state(i, states, adjacency_matrix)
    return new_states

def simulate_system(n, num_time_steps, initial_failures=None):
    """
    Simulates the system for a given number of time steps.

    Args:
        n: The number of nodes in the system.
        num_time_steps: The number of time steps to simulate.
        initial_failures: A list of node indices to initially set as failed (optional).

    Returns:
        A list of NumPy arrays, where each array represents the states of the nodes at each time step.
    """
    states, adjacency_matrix = initialize_system(n)

    if initial_failures:
        for node_index in initial_failures:
            states[node_index] = 1

    history = [states.copy()]  # Record the initial state

    for _ in range(num_time_steps):
        states = simulate_time_step(states, adjacency_matrix)
        history.append(states.copy())

    return history, adjacency_matrix

# Example usage with a 3x3 adjacency matrix:
num_nodes = 3
time_steps = 10

# Simulate with no initial failures:
history, adjacency_matrix = simulate_system(num_nodes, time_steps)

# Simulate with initial failures in nodes 0 and 2:
# history, adjacency_matrix = simulate_system(num_nodes, time_steps, initial_failures=[0, 2])

print("Adjacency Matrix:")
print(adjacency_matrix)

print("\nSystem State History:")
for i, states in enumerate(history):
    print(f"Time step {i}: {states}")