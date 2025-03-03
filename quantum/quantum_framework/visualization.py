# visualization.py
import matplotlib.pyplot as plt

def plot_failure(a_values, measured_failure, expected_failure, title='Failure Probability vs a'):
    plt.figure(figsize=(10, 6))
    plt.plot(a_values, measured_failure, 'o-', label='Measured Failure')
    plt.plot(a_values, expected_failure, 'k--', label='Expected Failure')
    plt.xlabel('a')
    plt.ylabel('Failure Probability')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()