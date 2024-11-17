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

def plot_real_times(times):
    """
    Plot real times from the file.
    Args:
        times (list): List of times in seconds.
    """
    x_values = list(range(len(times)))  # Position in the list as x-values
    
    plt.figure(figsize=(10, 6))
    
    # Plot times
    plt.scatter(x_values, times, color='blue', label='Real Times')
    
    # Graph details
    plt.title("Real Times from File")
    plt.xlabel("Index in List")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.grid(True)
    
    # Show plot
    plt.show()

# Example usage
file_path = "./out/slurm-last.out"

# Parse and plot real times from the single file
real_times = parse_real_times(file_path)
print("Real Times:", real_times)

# Plot the times
plot_real_times(real_times)
