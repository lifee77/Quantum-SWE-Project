import numpy as np

def initialize_system(adjacency_matrix, initial_states):
    """
    Initializes the system with a given adjacency matrix and initial states.

    Args:
        adjacency_matrix (np.ndarray): The adjacency matrix.
        initial_states (np.ndarray): The initial states of the nodes (binary: 0 or 1).

    Returns:
        tuple: (states, adjacency_matrix) where states is a 1D NumPy array of node states.
    """
    # Check that the adjacency matrix is square
    if adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
        raise ValueError("Adjacency matrix must be square")
    # Check that the number of states matches the size of the matrix
    if adjacency_matrix.shape[0] != len(initial_states):
        raise ValueError("Adjacency matrix dimensions must match the number of states")
    # Check that all elements are between 0 and 1
    if not np.all((adjacency_matrix >= 0) & (adjacency_matrix <= 1)):
        raise ValueError("Adjacency matrix elements must be between 0 and 1")
    # Use np.allclose for the diagonal check to allow for minor floating-point imprecision.
    if not np.allclose(np.diag(adjacency_matrix), 1):
        raise ValueError("Adjacency matrix diagonal elements must be 1")
    # Check that the initial states are binary (0 or 1)
    if not np.all((initial_states == 0) | (initial_states == 1)):
        raise ValueError("Initial states must be either 0 or 1")
    
    # Ensure that the states are of integer type
    return initial_states.astype(np.int8), adjacency_matrix.astype(np.float64)

def update_all_nodes(states, adjacency_matrix):
    """
    Vectorized update for all nodes in the system.

    For each healthy node (state 0), the probability of failure is calculated as:
        probability_of_failure = 1 - Π_j (1 - a_ij * state_j)
    Nodes already failed (state 1) remain failed.

    Args:
        states (np.ndarray): Current states of all nodes.
        adjacency_matrix (np.ndarray): The adjacency matrix.

    Returns:
        np.ndarray: The updated states after one time step.
    """
    # Compute the product for each node i over all j:
    # Each element (i,j) contributes (1 - a_ij * state_j). For a failed node j (state=1),
    # the factor becomes (1 - a_ij), and for a healthy node (state=0) it is 1.
    non_failure_prob = np.prod(1 - adjacency_matrix * states, axis=1)
    probability_of_failure = 1 - non_failure_prob

    # Generate random numbers for each node and determine which healthy nodes fail
    random_vals = np.random.rand(len(states))
    
    # Nodes that were already failed remain 1; healthy nodes become 1 if the random value is less than the failure probability.
    new_states = np.where(states == 1, 1, (random_vals < probability_of_failure).astype(np.int8))
    return new_states

def simulate_time_step(states, adjacency_matrix):
    """
    Simulates one synchronous time step for the entire system.

    Args:
        states (np.ndarray): The current states of all nodes.
        adjacency_matrix (np.ndarray): The adjacency matrix.

    Returns:
        np.ndarray: The updated states of the nodes after one time step.
    """
    return update_all_nodes(states, adjacency_matrix)

def simulate_system(adjacency_matrix, initial_states, num_time_steps, seed=None):
    """
    Simulates the system for a given number of time steps.

    Args:
        adjacency_matrix (np.ndarray): The adjacency matrix.
        initial_states (np.ndarray): The initial states of the nodes.
        num_time_steps (int): The number of time steps to simulate.
        seed (int, optional): A random seed for reproducibility.

    Returns:
        tuple: (history, adjacency_matrix) where history is a list of state arrays
               for each time step (including the initial state).
    """
    # Set random seed if provided for reproducibility
    if seed is not None:
        np.random.seed(seed)

    states, adjacency_matrix = initialize_system(adjacency_matrix, initial_states)
    history = [states.copy()]  # Record the initial state

    for _ in range(num_time_steps):
        states = simulate_time_step(states, adjacency_matrix)
        history.append(states.copy())

    return history, adjacency_matrix

# Example usage:
if __name__ == "__main__":
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

    # Simulate the system with a fixed random seed for reproducibility
    history, adj_matrix = simulate_system(adjacency_matrix, initial_states, time_steps, seed=42)

    print("Adjacency Matrix:")
    print(adj_matrix)

    print("\nSystem State History:")
    for i, states in enumerate(history):
        print(f"Time step {i}: {states}")
