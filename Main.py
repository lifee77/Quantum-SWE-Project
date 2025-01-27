import numpy as np
import random

def initialize_system(adjacency_matrix, initial_states):
    """
    Initializes the system with a given adjacency matrix and initial states.

    Args:
        adjacency_matrix: The adjacency matrix (NumPy array).
        initial_states: The initial states of the nodes (NumPy array).

    Returns:
        A tuple containing:
        - states: The initial states of the nodes.
        - adjacency_matrix: The adjacency matrix.
    """

    # Basic checks on input
    if adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
        raise ValueError("Adjacency matrix must be square")
    if adjacency_matrix.shape[0] != len(initial_states):
        raise ValueError("Adjacency matrix dimensions must match the number of states")
    if not np.all((adjacency_matrix >= 0) & (adjacency_matrix <= 1)):
        raise ValueError("Adjacency matrix elements must be between 0 and 1")
    if not np.all(np.diag(adjacency_matrix) == 1):
        raise ValueError("Adjacency matrix diagonal elements must be 1")
    if not np.all((initial_states == 0) | (initial_states == 1)):
        raise ValueError("Initial states must be either 0 or 1")

    return initial_states, adjacency_matrix

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

def simulate_system(adjacency_matrix, initial_states, num_time_steps):
    """
    Simulates the system for a given number of time steps.

    Args:
        adjacency_matrix: The adjacency matrix (NumPy array).
        initial_states: The initial states of the nodes (NumPy array).
        num_time_steps: The number of time steps to simulate.

    Returns:
        A list of NumPy arrays, where each array represents the states of the nodes at each time step.
    """
    states, adjacency_matrix = initialize_system(adjacency_matrix, initial_states)

    history = [states.copy()]  # Record the initial state

    for _ in range(num_time_steps):
        states = simulate_time_step(states, adjacency_matrix)
        history.append(states.copy())

    return history, adjacency_matrix

# Example usage:
num_nodes = 3
time_steps = 10

# Define a custom adjacency matrix
adjacency_matrix = np.array([
    [1.0, 0.9, 0.8],
    [0.7, 1.0, 0.6],
    [0.9, 0.85, 1.0]
])

# Define initial states (e.g., node 0 failed, others functioning)
initial_states = np.array([1, 0, 0])

# Simulate the system
history, adjacency_matrix = simulate_system(adjacency_matrix, initial_states, time_steps)

print("Adjacency Matrix:")
print(adjacency_matrix)

print("\nSystem State History:")
for i, states in enumerate(history):
    print(f"Time step {i}: {states}")