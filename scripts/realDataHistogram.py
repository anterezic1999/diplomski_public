import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import csv

def find_log_files(directory):
    return glob.glob(os.path.join(directory, "*.log"))

def parse_log_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data_start = next((i for i, line in enumerate(lines) if line.strip().startswith("|")), -1)
    if data_start == -1:
        return np.array([])

    data_lines = lines[data_start + 3:]

    jd = []

    for line in data_lines:
        try:
            values = line.split()
            if values:
                jd.append(float(values[0]))
        except ValueError:
            continue

    return np.array(jd)

def calculate_time_differences(jd):
    return np.diff(jd)

def process_files(directory):
    log_files = find_log_files(directory)
    all_time_diffs = []

    for file in log_files:
        jd = parse_log_file(file)
        if len(jd) > 1:
            time_diffs = calculate_time_differences(jd)
            time_diffs = time_diffs[time_diffs <= 365]
            all_time_diffs.extend(time_diffs)

    return np.array(all_time_diffs)

def plot_and_save_histogram(time_diffs, output_file):
    plt.figure(figsize=(12, 6))

    bins = list(range(101)) + [365]

    n, bins, patches = plt.hist(time_diffs, bins=bins, edgecolor='black')

    patches[-1].set_facecolor('blue')

    plt.xlabel('Time difference between successive measurements (days)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Time Differences (with 100-365 days grouped)')
    plt.grid(True, alpha=0.3)

    plt.xticks([0, 25, 50, 75, 100, 365])
    plt.text(232.5, max(n)/2, '100-365\ndays', ha='center', va='center')

    last_bin_count = n[-1]
    plt.text(232.5, last_bin_count, f'{int(last_bin_count)}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

    # Save histogram data to CSV
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Bin_Start', 'Bin_End', 'Count'])
        for i in range(len(n)):
            if i == len(n) - 1:
                writer.writerow([bins[i], bins[i+1], n[i]])
            else:
                writer.writerow([bins[i], bins[i+1], n[i]])

def print_statistics(time_diffs):
    print("\nStatistics of time differences (up to 365 days):")
    print(f"Minimum: {np.min(time_diffs):.2f} days")
    print(f"Maximum: {np.max(time_diffs):.2f} days")
    print(f"Mean: {np.mean(time_diffs):.2f} days")
    print(f"Median: {np.median(time_diffs):.2f} days")
    print(f"Standard deviation: {np.std(time_diffs):.2f} days")

    unique, counts = np.unique(np.round(time_diffs), return_counts=True)
    mode = unique[np.argmax(counts)]
    print(f"Mode (rounded to nearest day): {mode:.0f} days")

    percentiles = np.percentile(time_diffs, [25, 75])
    print(f"25th percentile: {percentiles[0]:.2f} days")
    print(f"75th percentile: {percentiles[1]:.2f} days")

    long_interval_count = np.sum(time_diffs > 100)
    print(f"Number of measurements with time difference > 100 days: {long_interval_count}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python script_name.py <directory_path>")
        sys.exit(1)

    directory = sys.argv[1]
    time_diffs = process_files(directory)

    if len(time_diffs) == 0:
        print("No valid data found in the log files.")
    else:
        output_file = 'histogram_data.csv'
        plot_and_save_histogram(time_diffs, output_file)
        print_statistics(time_diffs)
        print(f"\nHistogram data saved to {output_file}")
