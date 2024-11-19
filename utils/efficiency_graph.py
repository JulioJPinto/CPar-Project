import os
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

def average_real_times(folder_path):
    """
    Read all files in a folder, parse 'real' times, and compute the average for each position.
    Args:
        folder_path (str): Path to the folder containing files.
    Returns:
        list: List of averaged real times.
    """
    all_times = []

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            real_times = parse_real_times(file_path)
            
            # Extend the all_times list to accommodate the new data if necessary
            while len(all_times) < len(real_times):
                all_times.append([])

            # Add the parsed times to the corresponding positions in all_times
            for i, time in enumerate(real_times):
                all_times[i].append(time)

    # Compute the averages for each position
    averaged_times = [sum(times) / len(times) for times in all_times if times]
    return averaged_times

def plot_efficiency(times):
    """
    Plot efficiency with ticks for each integer value on both axes.
    Args:
        times (list): List of times in seconds.
    """
    if not times or times[0] == 0:
        raise ValueError("The baseline time (first entry) cannot be zero or empty.")

    baseline = times[0]  # Use the first time as the baseline
    speedups = [baseline / t for t in times]  # Compute speedup
    processors = list(range(1, len(speedups) + 1))  # Number of processors (threads)

    # Compute efficiency
    efficiencies = [speedup / p for speedup, p in zip(speedups, processors)]

    plt.figure(figsize=(12, 8))

    # Plot efficiency
    plt.scatter(processors, efficiencies, color='blue', label='Efficiency')

    # Set ticks for each integer value
    plt.xticks(processors, fontsize=10)
    plt.yticks(fontsize=10)

    # Graph details
    plt.title("Efficiency - Number of Threads", fontsize=16)
    plt.xlabel("Threads", fontsize=14)
    plt.ylabel("Efficiency", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Show plot
    plt.show()

# Example usage
folder_path = "./out/"  # Path to the folder containing files

# Parse and average real times from all files in the folder
averaged_times = average_real_times(folder_path)
print("Averaged Real Times:", averaged_times)

# Plot the efficiency based on the averaged times
plot_efficiency(averaged_times)
