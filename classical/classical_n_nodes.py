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
    # Allow for floating point tolerance when checking the diagonal
    if not np.allclose(np.diag(adjacency_matrix), 1):
        raise ValueError("Adjacency matrix diagonal elements must be 1")
    # Check that the initial states are binary (0 or 1)
    if not np.all((initial_states == 0) | (initial_states == 1)):
        raise ValueError("Initial states must be either 0 or 1")
    
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
    # For each node i, compute the product over all j of (1 - a_ij * state_j).
    non_failure_prob = np.prod(1 - adjacency_matrix * states, axis=1)
    probability_of_failure = 1 - non_failure_prob

    # Generate a random number for each node and decide which healthy nodes fail.
    random_vals = np.random.rand(len(states))
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

# --- Example 1: Using n = 10 Nodes ---

def example_n_10():
    n = 10
    time_steps = 10

    # Generate a random adjacency matrix for n nodes.
    # Off-diagonals are random values between 0 and 1.
    adjacency_matrix = np.random.rand(n, n)
    np.fill_diagonal(adjacency_matrix, 1.0)  # Ensure the diagonal is 1.

    # Create random initial states for n nodes (e.g., 20% chance of being failed)
    initial_states = np.random.choice([0, 1], size=n, p=[0.8, 0.2])

    # Simulate the system
    history, adj_matrix = simulate_system(adjacency_matrix, initial_states, time_steps, seed=42)

    print("Example with n = 10 Nodes:")
    print("Adjacency Matrix:")
    print(adj_matrix)
    print("\nInitial States:")
    print(initial_states)
    print("\nSystem State History:")
    for i, states in enumerate(history):
        print(f"Time step {i}: {states}")
    print("\n" + "="*50 + "\n")

# --- Example 2: Using n = 50 Nodes ---

def example_n_50():
    n = 50
    time_steps = 15

    # Generate a random adjacency matrix for n nodes.
    adjacency_matrix = np.random.rand(n, n)
    np.fill_diagonal(adjacency_matrix, 1.0)  # Diagonal must be 1.

    # Create random initial states for n nodes (e.g., 10% chance of being failed)
    initial_states = np.random.choice([0, 1], size=n, p=[0.9, 0.1])

    # Simulate the system
    history, adj_matrix = simulate_system(adjacency_matrix, initial_states, time_steps, seed=123)

    print("Example with n = 50 Nodes:")
    print("Adjacency Matrix (showing first 5 rows):")
    print(adj_matrix[:5])  # For brevity, print only the first 5 rows
    print("\nInitial States:")
    print(initial_states)
    print("\nSystem State History:")
    for i, states in enumerate(history):
        print(f"Time step {i}: {states}")
    print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    example_n_10()
    example_n_50()
