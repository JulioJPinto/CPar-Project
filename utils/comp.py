import re
import matplotlib.pyplot as plt

def parse_real_times(file_path):
    """
    Parse 'real' times from the given file.
    Args:
        file_path (str): Path to the input file.
    Returns:
        list: List of real times in seconds.
    """
    real_times = []
    time_pattern = re.compile(r"real\s+(\d+)m([\d.]+)s")
    
    with open(file_path, 'r') as file:
        for line in file:
            match = time_pattern.search(line)
            if match:
                minutes = int(match.group(1))
                seconds = float(match.group(2))
                total_seconds = minutes * 60 + seconds
                real_times.append(total_seconds)
    
    return real_times

def compare_real_times(file1, file2):
    """
    Compare real times between two files.
    Args:
        file1 (str): Path to the first file.
        file2 (str): Path to the second file.
    Returns:
        dict: Comparison results including lists of times and differences.
    """
    times1 = parse_real_times(file1)
    times2 = parse_real_times(file2)
    
    comparison = {
        "file1_times": times1,
        "file2_times": times2,
        "differences": [t1 - t2 for t1, t2 in zip(times1, times2)]
    }
    return comparison

def plot_real_times(times1, times2):
    """
    Plot real times from both files.
    Args:
        times1 (list): List of times from file 1.
        times2 (list): List of times from file 2.
    """
    x_values = list(range(len(times1)))  # Position in the list as x-values
    
    plt.figure(figsize=(10, 6))
    
    # Plot File 1 times
    plt.scatter(x_values, times1, color='blue', label='File 1 Times')
    
    # Plot File 2 times
    plt.scatter(x_values, times2, color='red', label='File 2 Times')
    
    # Graph details
    plt.title("Comparison of Real Times")
    plt.xlabel("Index in List")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.grid(True)
    
    # Show plot
    plt.show()

# Example usage
file1_path = "./out/slurm-last.out"
file2_path = "./out/slurm-test.out"

comparison_results = compare_real_times(file1_path, file2_path)
print("File 1 Times:", comparison_results["file1_times"])
print("File 2 Times:", comparison_results["file2_times"])
print("Differences:", comparison_results["differences"])

# Plot the times
plot_real_times(comparison_results["file1_times"], comparison_results["file2_times"])
