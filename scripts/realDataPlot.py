import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from astropy.timeseries import LombScargle
import sys
from collections import defaultdict

def find_log_files(directory):
    return glob.glob(os.path.join(directory, "*.log"))

def parse_log_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    star_name = next((line.split('=')[1].strip().strip('"').strip("'") for line in lines if "STAR_ID" in line), "Unknown")

    data_start = next(i for i, line in enumerate(lines) if line.startswith("|"))
    data_lines = lines[data_start + 3:]

    jd = []
    rv = []
    rv_uncertainty = []

    for line in data_lines:
        if line.strip():
            values = line.split()
            try:
                if len(values) >= 3:
                    jd_val = float(values[0])
                    rv_val = float(values[1])
                    rv_unc_val = float(values[2])
                    jd.append(jd_val)
                    rv.append(rv_val)
                    rv_uncertainty.append(rv_unc_val)
            except ValueError:
                continue

    return np.array(jd), np.array(rv), np.array(rv_uncertainty), star_name

def generate_periodogram(time_array, rv):
    if np.any(np.isnan(time_array)) or np.any(np.isnan(rv)):
        print("Warning: NaN values in input data")
        return None, None

    ls = LombScargle(time_array, rv)

    min_period = 6 / 24
    growth_factor = 1.0102925678

    exponents = np.arange(1000)
    periods = min_period * (growth_factor ** exponents)

    frequencies = 1 / periods

    power = ls.power(frequencies, normalization='psd')

    if np.any(np.isnan(power)):
        print("Warning: NaN values in power spectrum")
        return None, None

    return frequencies, power

def plot_periodogram(jd, rv, star_name):
    f, Pxx = generate_periodogram(jd, rv)
    plt.figure(figsize=(12, 6))
    plt.plot(1/f, Pxx)
    plt.xlabel('Period (days)')
    plt.ylabel('Power')
    plt.title(f'Periodogram for {star_name}')
    plt.xscale('log')

    min_period = min(1/f)
    max_period = max(1/f)

    plt.gca().invert_xaxis()
    plt.xlim(max_period, min_period)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

    plt.tight_layout()
    plt.show()
    plt.close()

def plot_data(jd, rv, rv_uncertainty, star_name):
    if len(jd) == 0:
        print("No valid data points found.")
        return

    print(f"Number of data points: {len(jd)}")
    print(f"JD range: {jd.min()} to {jd.max()} -> {jd.max() - jd.min()}")
    print(f"RV range: {rv.min()} to {rv.max()} m/s")

    time_since_first = jd - jd[0]

    plt.figure(figsize=(10, 6))
    plt.errorbar(time_since_first, rv, yerr=rv_uncertainty, fmt='o', capsize=5, ecolor='red', markersize=4)
    plt.xlabel('Time since first measurement (days)')
    plt.ylabel('Radial Velocity (m/s)')
    plt.title(f'Radial Velocity Curve for {star_name}')
    plt.grid(True)
    plt.show()

# Main script
if len(sys.argv) != 2:
    print("Usage: python script_name.py <directory_path>")
    sys.exit(1)

directory = sys.argv[1]
log_files = find_log_files(directory)

if not log_files:
    print("No .log files found in the specified directory.")
else:
    # Group files by star name
    grouped_files = defaultdict(list)
    for file in log_files:
        _, rv, _, star_name = parse_log_file(file)
        grouped_files[star_name].append(file)

    # Sort the groups
    sorted_groups = sorted(grouped_files.items())

    print(f"Number of unique systems: {len(sorted_groups)}")

    # Count stars split across different IDs
    split_stars = 0
    for star_name, files in sorted_groups:
        ids = set()
        for file in files:
            file_name = os.path.basename(file)
            parts = file_name.split('_')
            if len(parts) >= 2:
                ids.add(parts[1])
        if len(ids) > 1:
            split_stars += 1

    print(f"Number of stars split across different IDs: {split_stars}")

    # Print star names and their indexes
    print("\nStar Systems:")
    for index, (star_name, _) in enumerate(sorted_groups):
        print(f"{index}: {star_name}")

    while True:
        choice = input("\nEnter the index of the system you want to analyze (or 'q' to quit): ")

        if choice.lower() == 'q':
            break

        try:
            system_index = int(choice)
            if 0 <= system_index < len(sorted_groups):
                star_name, files = sorted_groups[system_index]

                all_jd = []
                all_rv = []
                all_rv_uncertainty = []

                for file in files:
                    jd, rv, rv_uncertainty, _ = parse_log_file(file)
                    all_jd.extend(jd)
                    all_rv.extend(rv)
                    all_rv_uncertainty.extend(rv_uncertainty)

                all_jd = np.array(all_jd)
                all_rv = np.array(all_rv)
                all_rv_uncertainty = np.array(all_rv_uncertainty)

                # Sort the data by JD
                sort_indices = np.argsort(all_jd)
                all_jd = all_jd[sort_indices]
                all_rv = all_rv[sort_indices]
                all_rv_uncertainty = all_rv_uncertainty[sort_indices]

                mean_rv = int(np.mean(all_rv))
                all_rv = all_rv - mean_rv

                plot_data(all_jd, all_rv, all_rv_uncertainty, star_name)
                plot_periodogram(all_jd, all_rv, star_name)
            else:
                print("Invalid index. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number or 'q' to quit.")
