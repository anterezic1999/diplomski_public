import os
import glob
import numpy as np
import math
import h5py
from astropy.timeseries import LombScargle
from tqdm import tqdm
import re
import pandas as pd

def find_log_files(directory):
    return glob.glob(os.path.join(directory, "*.log"))

def parse_log_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

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

    return np.array(jd), np.array(rv), np.array(rv_uncertainty)

def load_star_masses(csv_path):
    df = pd.read_csv(csv_path)
    return dict(zip(df['name'], df['mass']))

def extract_star_name(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            if 'STAR_ID' in line:
                # Try matching the format with double quotes
                match = re.search(r'STAR_ID\s*=\s*"(.+?)"', line)
                if match:
                    return match.group(1)

                # If not found, try matching the format with single quotes
                match = re.search(r"STAR_ID\s*=\s*'(.+?)'", line)
                if match:
                    return match.group(1)
    return None

def calculate_periodogram(time, rv, num_peaks=10):
    frequency, power = LombScargle(time, rv).autopower()
    sorted_indices = np.argsort(power)[::-1]
    top_frequencies = frequency[sorted_indices][:num_peaks]
    top_powers = power[sorted_indices][:num_peaks]
    return np.concatenate((top_frequencies, top_powers))

def process_star_system(file_paths):
    all_jd = []
    all_rv = []
    all_rv_uncertainty = []

    for file_path in file_paths:
        jd, rv, rv_uncertainty = parse_log_file(file_path)
        all_jd.extend(jd)
        all_rv.extend(rv)
        all_rv_uncertainty.extend(rv_uncertainty)

    all_jd = np.array(all_jd)
    all_rv = np.array(all_rv)
    all_rv_uncertainty = np.array(all_rv_uncertainty)

    # Sort all data by JD
    sort_indices = np.argsort(all_jd)
    all_jd = all_jd[sort_indices]
    all_rv = all_rv[sort_indices]
    all_rv_uncertainty = all_rv_uncertainty[sort_indices]

    periodogram_rv = calculate_periodogram(all_jd, all_rv)
    periodogram_uncertainties = calculate_periodogram(all_jd, all_rv_uncertainty)

    return all_jd, all_rv, all_rv_uncertainty, periodogram_rv, periodogram_uncertainties

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    input_dir = os.path.join(project_root, 'real_data')
    output_dir = os.path.join(project_root, 'data')
    output_file = os.path.join(output_dir, 'real_data.h5')
    star_masses_file = os.path.join(output_dir, 'star_masses.csv')

    stats = []
    lenOfObservation = []

    os.makedirs(output_dir, exist_ok=True)

    print(f"Input directory: {input_dir}")
    print(f"Output file: {output_file}")

    # Load star masses
    star_masses = load_star_masses(star_masses_file)
    print(f"Loaded {len(star_masses)} star masses")

    log_files = find_log_files(input_dir)
    print(f"Number of log files found: {len(log_files)}")

    # Group files by star system
    star_systems = {}
    for file_path in log_files:
        file_name = os.path.basename(file_path)
        match = re.match(r'(.+)_\d{3}\.log', file_name)
        if match:
            system_name = match.group(1)
            if system_name not in star_systems:
                star_systems[system_name] = []
            star_systems[system_name].append(file_path)

    print(f"Number of unique star systems: {len(star_systems)}")

    with h5py.File(output_file, 'w') as hf:
        for system_name, file_paths in tqdm(star_systems.items(), desc="Processing star systems"):
            try:
                jd, rv, rv_uncertainty, periodogram_rv, periodogram_uncertainties = process_star_system(file_paths)

                # Extract star name from the first file of the system
                star_name = extract_star_name(file_paths[0])

                # Get star mass from the loaded data, or use default value of 1
                star_mass = star_masses.get(star_name, 1.0)
                if math.isnan(star_mass):
                    #print(f"No mass data found for {star_name}, using default value of 1.0")
                    star_mass = 1.0
                star_mass = star_mass * 1.989e30
                #print(f"Star: {star_name}, Mass: {star_mass}")

                # Create a group for this star system
                group = hf.create_group(system_name)

                # Store star mass and planet count
                group.attrs['star_mass'] = star_mass
                group.attrs['planet_count'] = 1
                group.attrs['star_name'] = star_name

                # Store radial velocity data
                group.create_dataset('time', data=jd)
                lenOfObservation.append(max(jd) - min(jd))
                stats.append(len(jd))
                group.create_dataset('radial_velocity', data=rv)
                group.create_dataset('uncertainties_data', data=rv_uncertainty)
                group.create_dataset('periodogram_rv', data=periodogram_rv)
                group.create_dataset('periodogram_uncertainties', data=periodogram_uncertainties)

                # Store planet data (dummy values)
                planets = group.create_group('planets')
                planet_group = planets.create_group('planet_0')
                planet_group.attrs['mass'] = 1.0
                planet_group.attrs['semi_major_axis'] = 1.0
                planet_group.attrs['eccentricity'] = 0.0
                planet_group.attrs['visibility'] = 1.0

            except Exception as e:
                print(f"Error processing star system {system_name}: {e}")
                continue

    print(f"Data processing complete. Output saved to {output_file}")
    print(f"Output file size: {os.path.getsize(output_file)} bytes")
    print(f"Mean of count of observations: {np.mean(stats):.4f}")
    print(f"Median of count of observations: {np.median(stats):.4f}")
    print(f"Standard deviation of count of observations: {np.std(stats):.4f}")
    print(f"Minimum of count of observations: {np.min(stats):.4f}")
    print(f"Maximum of count of observations: {np.max(stats):.4f}")

    print(f"Mean of len of observations: {np.mean(lenOfObservation):.4f}")
    print(f"Median of len of observations: {np.median(lenOfObservation):.4f}")
    print(f"Standard deviation of len of observations: {np.std(lenOfObservation):.4f}")
    print(f"Minimum of len of observations: {np.min(lenOfObservation):.4f}")
    print(f"Maximum of len of observations: {np.max(lenOfObservation):.4f}")

if __name__ == "__main__":
    main()
