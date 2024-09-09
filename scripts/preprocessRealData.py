import os
import glob
import numpy as np
from astropy.timeseries import LombScargle
import sys
from collections import defaultdict
import h5py

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
                    if not (np.isnan(jd_val) or np.isnan(rv_val) or np.isnan(rv_unc_val)):
                        jd.append(jd_val)
                        rv.append(rv_val)
                        rv_uncertainty.append(rv_unc_val)
            except ValueError:
                continue  # Skip lines that can't be converted to float

    return np.array(jd), np.array(rv), np.array(rv_uncertainty), star_name

def generate_periodogram(time_array, rv):
    if len(time_array) == 0 or len(rv) == 0:
        print("Warning: Empty input data")
        return None, None

    ls = LombScargle(time_array, rv)

    min_period = 6 / 24  # 6 hours in days
    growth_factor = 1.0102925678

    exponents = np.arange(1000)
    periods = min_period * (growth_factor ** exponents)

    frequencies = 1 / periods

    power = ls.power(frequencies, normalization='psd')

    return frequencies, power

def save_periodogram_to_hdf5(output_path, star_name, frequencies, power):
    with h5py.File(output_path, 'a') as hf:
        group = hf.create_group(star_name)
        group.create_dataset('frequencies', data=frequencies)
        group.create_dataset('power', data=power)

def process_star_data(files, output_path):
    all_jd = []
    all_rv = []
    all_rv_uncertainty = []
    star_name = None

    for file in files:
        jd, rv, rv_uncertainty, file_star_name = parse_log_file(file)
        all_jd.extend(jd)
        all_rv.extend(rv)
        all_rv_uncertainty.extend(rv_uncertainty)
        if star_name is None:
            star_name = file_star_name

    all_jd = np.array(all_jd)
    all_rv = np.array(all_rv)
    all_rv_uncertainty = np.array(all_rv_uncertainty)

    if len(all_jd) == 0:
        print(f"No valid data points for {star_name}")
        return

    sort_indices = np.argsort(all_jd)
    all_jd = all_jd[sort_indices]
    all_rv = all_rv[sort_indices]
    all_rv_uncertainty = all_rv_uncertainty[sort_indices]

    mean_rv = int(np.mean(all_rv))
    all_rv = all_rv - mean_rv

    frequencies, power = generate_periodogram(all_jd, all_rv)

    if frequencies is not None and power is not None:
        save_periodogram_to_hdf5(output_path, star_name, frequencies, power)
    else:
        print(f"Failed to generate periodogram for {star_name}")

def main(input_directory, output_path):
    log_files = find_log_files(input_directory)

    if not log_files:
        print("No .log files found in the specified directory.")
        return

    grouped_files = defaultdict(list)
    for file in log_files:
        _, _, _, star_name = parse_log_file(file)
        grouped_files[star_name].append(file)

    sorted_groups = sorted(grouped_files.items())

    print(f"Number of unique systems: {len(sorted_groups)}")

    for star_name, files in sorted_groups:
        process_star_data(files, output_path)

    print(f"All periodograms saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_directory> <output_file.h5>")
        sys.exit(1)

    input_directory = sys.argv[1]
    output_path = sys.argv[2]

    main(input_directory, output_path)
