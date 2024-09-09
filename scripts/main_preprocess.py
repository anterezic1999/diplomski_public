import torch
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import periodogram
from astropy.timeseries import LombScargle
from tqdm import tqdm
import numpy as np
from scipy.stats import gamma
import argparse
import h5py
import random
import multiprocessing
from functools import partial
import string
import csv

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PERIODOGRAM_LEN=1000

def load_histogram_data(file_path):
    bins = []
    counts = []
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            bins.append(float(row[0]))
            counts.append(int(round(float(row[2]))))
    return np.array(bins), np.array(counts)

def create_time_sampler(bins, counts):
    cdf = np.cumsum(counts)
    cdf = cdf / cdf[-1]

    def sample_time_interval(num_samples=1):
        random_values = np.random.random(num_samples)
        delta_times = np.interp(random_values, cdf, bins)

        # Handle the last bin (100-365 days) separately
        last_bin_mask = delta_times == bins[-2]
        last_bin_samples = np.sum(last_bin_mask)
        if last_bin_samples > 0:
            delta_times[last_bin_mask] = np.random.uniform(100, 365, last_bin_samples)

        # For all other bins, sample uniformly within the bin
        for i in range(len(bins) - 1):
            bin_mask = (delta_times >= bins[i]) & (delta_times < bins[i + 1])
            bin_samples = np.sum(bin_mask)
            if bin_samples > 0:
                delta_times[bin_mask] = np.random.uniform(bins[i], bins[i + 1], bin_samples)

        return delta_times

    return sample_time_interval

def load_single_system(filename, directory):
    with open(os.path.join(directory, filename), 'r') as file:
        return json.load(file)

def load_system_data_in_chunks(directory, chunk_size, max_files, num_processes=None):
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')][:max_files]

    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    load_func = partial(load_single_system, directory=directory)

    with multiprocessing.Pool(processes=num_processes) as pool:
        for i in range(0, len(json_files), chunk_size):
            chunk = json_files[i:i+chunk_size]
            systems = list(tqdm(
                pool.imap(load_func, chunk),
                total=len(chunk),
                desc=f"Loading system data (chunk {i//chunk_size + 1})",
                unit="file"
            ))
            yield systems

def calculate_K(mass_planet, period, e, mass_star, device):
    G = 6.67430e-11
    G_tensor = torch.tensor(G, dtype=torch.float32, device=device)
    period_sec = period * 86400
    K = (2 * torch.pi * G_tensor / period_sec) ** (1 / 3) * (
            mass_planet / (mass_star + mass_planet) ** (2 / 3)) / torch.sqrt(1 - e ** 2)
    return K

def calculate_rv(system, time_tensor, device):
    mass_star = torch.tensor(system['star_mass'], dtype=torch.float32, device=device)
    rv_total = torch.zeros_like(time_tensor)

    for planet in system['planets']:
        mass = torch.tensor(planet['mass'], dtype=torch.float32, device=device)
        P_days = torch.tensor(planet['P'], dtype=torch.float32, device=device)
        e = torch.tensor(planet['e'], dtype=torch.float32, device=device)
        w = torch.tensor(planet['w'], dtype=torch.float32, device=device)
        phase_offset = torch.tensor(planet.get('phase_offset', 0), dtype=torch.float32, device=device)

        M = 2 * torch.pi * time_tensor / P_days + phase_offset
        E = M + e * torch.sin(M)

        theta = 2 * torch.atan2(torch.sqrt(1 + e) * torch.sin(E / 2), torch.sqrt(1 - e) * torch.cos(E / 2))

        K = calculate_K(mass, P_days, e, mass_star, device)
        rv = K * (torch.cos(theta + w) + e * torch.cos(w))
        rv_total += rv

    return rv_total

def quasi_periodic_kernel_gpu(t, A, tau, epsilon, P):
    diff = t.unsqueeze(1) - t.unsqueeze(0)
    return A**2 * torch.exp(-(diff**2) / (2*tau**2) - 2/epsilon * torch.sin(torch.pi*diff/P)**2)

def generate_stellar_noise_gpu(time_tensor, device):
    t = time_tensor.to(device)
    n_points = len(t)
    time_span = t[-1] - t[0]

    # Check if time array is strictly increasing
    if not torch.all(t[1:] > t[:-1]):
        #print("Time array is not strictly increasing")
        # Force strict increase by adding a small increment
        t = torch.cumsum(torch.abs(torch.diff(t, prepend=torch.tensor([0.0], device=device))), dim=0)

    mean_uncertainty = 1.0
    uncertainties = torch.normal(mean_uncertainty, 0.3, (n_points,), device=device)
    uncertainties = torch.clamp(uncertainties, min=0.5 * mean_uncertainty)
    intrinsic_errors = torch.normal(0, uncertainties)

    freqs = torch.linspace(1/time_span, 10, n_points, device=device)
    pulsation_power = 1 / (1 + (freqs/0.1)**2)
    granulation_power = 10 / (1 + (freqs/0.01)**2)
    total_power = pulsation_power + granulation_power
    pulsation_granulation = torch.normal(0, torch.sqrt(total_power))

    P = torch.rand(1, device=device) * 30 + 10  # Uniform between 10 and 40
    A = torch.distributions.Gamma(torch.tensor([2.0]), torch.tensor([0.5])).sample().to(device)
    tau = torch.clamp(torch.normal(3*P, 0.1*P), min=P*0.1)  # Ensure tau is positive and not too small
    epsilon = torch.rand(1, device=device) * 0.5 + 0.5  # Uniform between 0.5 and 1.0

    K = quasi_periodic_kernel_gpu(t, A, tau, epsilon, P)

    # Scale the matrix to improve numerical stability
    scale = torch.max(torch.abs(K))
    K_scaled = K / scale

    # Add a larger regularization term
    K_scaled += torch.eye(n_points, device=device) * 1e-3

    try:
        L = torch.linalg.cholesky(K_scaled)
        rotational_modulation = torch.matmul(L, torch.randn(n_points, device=device)) * torch.sqrt(scale)
    except RuntimeError:
        print("Warning: Cholesky decomposition failed. Using diagonal approximation.")
        rotational_modulation = torch.normal(0, torch.sqrt(torch.diag(K)))

    total_noise = intrinsic_errors + pulsation_granulation + rotational_modulation
    return total_noise.cpu().numpy()

def generate_periodogram(time_array, rv):
    if np.any(np.isnan(time_array)) or np.any(np.isnan(rv)):
        print("Warning: NaN values in input data")
        return None, None

    ls = LombScargle(time_array, rv)

    # Set the minimum period (6 hours in days) and the growth factor
    min_period = 6 / 24  # 6 hours in days
    growth_factor = 1.0102925678

    # Generate 1000 periods, starting from min_period and increasing by the growth factor each time
    exponents = np.arange(1000)
    periods = min_period * (growth_factor ** exponents)

    # Convert periods to frequencies
    frequencies = 1 / periods

    power = ls.power(frequencies, normalization='psd')

    if np.any(np.isnan(power)):
        #print("NaN values in power spectrum")
        return None, None

    return frequencies, power

def remove_signal(time_array, rv, period):
    omega = 2 * np.pi / period
    A = np.vstack([np.cos(omega * time_array), np.sin(omega * time_array), np.ones_like(time_array)]).T
    coeffs, _, _, _ = np.linalg.lstsq(A, rv, rcond=None)
    fitted_signal = coeffs[0] * np.cos(omega * time_array) + coeffs[1] * np.sin(omega * time_array) + coeffs[2]
    return rv - fitted_signal

def get_top_peak(f, Pxx):
    max_power_index = np.argmax(Pxx)
    return 1/f[max_power_index], Pxx[max_power_index], max_power_index

def remove_frequency(f, Pxx, peak_index, width=1):
    # Create a notch filter
    notch = np.ones_like(Pxx)
    notch[max(0, peak_index-width):min(len(Pxx), peak_index+width+1)] = 0

    # Apply the notch filter
    Pxx_filtered = Pxx * notch

    return f, Pxx_filtered

def get_peak_padded(peak_index, f, Pxx, n_points=30):
    total_points = 2 * n_points + 1  # Total points including the center

    # Calculate start and end indices
    start_index = max(0, peak_index - n_points)
    end_index = min(len(f), peak_index + n_points + 1)

    # Initialize array with zeros
    padded_Pxx = np.zeros(total_points)

    # Fill in the actual values
    actual_start = n_points - (peak_index - start_index)
    actual_end = actual_start + (end_index - start_index)

    padded_Pxx[actual_start:actual_end] = Pxx[start_index:end_index]

    # Normalize padded_Pxx
    normalized_Pxx = (padded_Pxx - np.min(padded_Pxx)) / (np.max(padded_Pxx) - np.min(padded_Pxx))

    return normalized_Pxx

def pad_truncate(array, target_length=PERIODOGRAM_LEN):
    current_length = len(array)
    if current_length > target_length:
        # Truncate
        return array[:target_length]
    elif current_length < target_length:
        # Pad with zeros
        return np.pad(array, (0, target_length - current_length), 'constant')
    else:
        # Already the correct length
        return array
def scatter_plot_rv_data(total_rv, time_array):
    plt.figure(figsize=(12, 8))
    plt.scatter(time_array, total_rv, s=10)
    plt.xlabel('Time (Days)')
    plt.ylabel('Radial Velocity (m/s)')
    plt.title('Radial Velocity vs Time')
    plt.grid()
    plt.show()

def generate_time_span():
    mean = 1784.78
    std_dev = 1567.18
    min_span = 10
    max_span = 7000

    while True:
        span = random.gauss(mean, std_dev)
        if min_span <= span <= max_span:
            return span
def process_systems_chunk(systems, time_span_max, observation_entropy, time_sampler, device='cuda' if torch.cuda.is_available() else 'cpu'):
    processed_data = []

    for system in tqdm(systems, desc="Processing systems", unit="system"):
        planet_data = []
        non_planet_data = []

        if time_span_max == 0:
            time_span = generate_time_span()
        else:
            time_span = time_span_max

        # Generate time array based on the ratio
        ratio = random.uniform(0, observation_entropy)
        time_array = [0]
        while time_array[-1] < time_span:
            if random.random() < ratio:
                # Use time sampler
                delta_time = time_sampler(1)[0]
            else:
                # Use 1 day interval
                delta_time = 1 * random.uniform(0.95, 1.05)
            time_array.append(time_array[-1] + delta_time)
        time_array = np.array(time_array[:-1])  # Remove the last point if it exceeds time_span

        time_tensor = torch.tensor(time_array, dtype=torch.float32, device=device)

        stellar_noise = generate_stellar_noise_gpu(time_tensor, device)

        rv = calculate_rv(system, time_tensor, device)
        total_rv = rv.cpu().numpy() + stellar_noise

        #scatter_plot_rv_data(total_rv, time_array)

        if np.any(np.isnan(time_array)) or np.any(np.isnan(total_rv)):
            print(f"Warning: NaN values in time_array or total_rv for system")
            continue

        if len(time_array) != len(total_rv):
            print(f"Warning: Mismatch in lengths of time_array and total_rv for system")
            continue

        f, Pxx = generate_periodogram(time_array, total_rv)

        if f is None or Pxx is None or np.any(np.isnan(f)) or np.any(np.isnan(Pxx)):
            #print(f"Could not generate valid periodogram for system.")
            continue

        original_Pxx = Pxx
        Pxx = (Pxx - np.min(Pxx))/np.max(Pxx)

        planet_periods = [planet['P'] for planet in system['planets']]
        planet_masses = [planet['mass']/5.972e24 for planet in system['planets']]
        planet_frequencies = [1/period for period in planet_periods]

        # Get all peaks
        all_peaks = get_all_peaks(f, Pxx)

        if not all_peaks:
            # If no peaks are found, skip this system
            continue

        all_peaks.sort(key=lambda x: x[1], reverse=True)  # Sort by power, descending

        if planet_frequencies:  # If the system has planets
            # Process planet data
            for freq in planet_frequencies:
                if all_peaks:
                    closest_peak = min(all_peaks, key=lambda x: abs(1/x[0] - freq))
                    if abs(1/closest_peak[0] - freq) / freq < 0.1:  # 10% tolerance
                        planet_data.append({
                            'peak_frequency': 1/closest_peak[0],
                            'peak_power': closest_peak[1],
                            "focused_peak": get_peak_padded(closest_peak[2], f, Pxx),
                            'true_frequency': freq,
                            'label': 1
                        })

            # Process non-planet data
            num_non_planets = len(planet_data)
            high_power_count = num_non_planets * 3
            random_count = int(high_power_count / 2)

            # Select high power non-planet peaks
            high_power_peaks = [peak for peak in all_peaks
                                if all(abs(1/peak[0] - freq) / freq >= 0.1 for freq in planet_frequencies)][:high_power_count]

            for peak in high_power_peaks:
                non_planet_data.append({
                    'peak_frequency': 1/peak[0],
                    'peak_power': peak[1],
                    "focused_peak": get_peak_padded(peak[2], f, Pxx),
                    'label': 0
                })

            # Select random non-planet peaks
            remaining_peaks = [peak for peak in all_peaks
                               if peak not in high_power_peaks and
                               all(abs(1/peak[0] - freq) / freq >= 0.1 for freq in planet_frequencies)]
            random_peaks = random.sample(remaining_peaks, min(random_count, len(remaining_peaks)))

            for peak in random_peaks:
                non_planet_data.append({
                    'peak_frequency': 1/peak[0],
                    'peak_power': peak[1],
                    "focused_peak": get_peak_padded(peak[2], f, Pxx),
                    'label': 0
                })

        else:  # If the system has no planets
            # Select top 4 power frequencies or all if less than 4
            top_peaks = all_peaks[:min(4, len(all_peaks))]
            for peak in top_peaks:
                non_planet_data.append({
                    'peak_frequency': 1/peak[0],
                    'peak_power': peak[1],
                    "focused_peak": get_peak_padded(peak[2], f, Pxx),
                    'label': 0
                })

        system_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
        # Only add the system if we have any data
        if planet_data or non_planet_data:
            #print(np.min(Pxx), np.max(Pxx))
            processed_data.append({
                'name': system_name,
                "periodogram_f": f,
                "periodogram_Pxx": Pxx,
                'planet_data': planet_data,
                'non_planet_data': non_planet_data,
                "observation_duration": time_span,
                'original_system': system
            })

        # plot_periodogram(f, Pxx, system_name, planet_frequencies, planet_masses)
        #plot_periodogram(f, original_Pxx, system_name, planet_frequencies, planet_masses)

    return processed_data



def plot_periodogram(f, Pxx, system_name, planet_frequencies, planet_masses):
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(1/f, Pxx, zorder=3, lw=1)
    plt.xlabel('Period (days)')
    plt.ylabel('Power')
    plt.title(f' ')
    plt.xscale('log')

    # Get the period range of the periodogram
    min_period = min(1/f)
    max_period = max(1/f)

    # Filter planet frequencies to only those within the periodogram's range
    valid_planet_frequencies = [freq for freq in planet_frequencies if min_period <= 1/freq <= max_period]

    # Add vertical lines for valid planet periods
    for freq in valid_planet_frequencies:
        plt.axvline(x=1/freq, color='r', linestyle='--', alpha=0.5)

    # Annotate valid planet periods
    for i, freq in enumerate(valid_planet_frequencies):
        plt.text(1/freq, plt.ylim()[1], f'P{i+1} - M:{planet_masses[i]:.2f}', rotation=90, va='bottom')

    # Reverse x-axis so shorter periods are on the left
    plt.gca().invert_xaxis()

    # Set x-axis limits to show relevant period range
    plt.xlim(max_period, min_period)

    # Format y-axis to use scientific notation
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

    plt.tight_layout()
    plt.show()

    plt.close()

def get_all_peaks(f, Pxx):
    peaks = []
    masked_indices = set()

    while True:
        # Find the most powerful frequency that's not masked
        available_indices = [i for i in range(len(Pxx)) if i not in masked_indices]
        if not available_indices:
            break

        peak_idx = max(available_indices, key=lambda i: Pxx[i])
        peak_power = Pxx[peak_idx]

        # Check if we should terminate
        if peak_power < np.percentile(Pxx, 50) * 3:
            break

        peaks.append((1/f[peak_idx], peak_power, peak_idx))

        # Mask the peak and its surrounding frequencies
        peak_indices = [peak_idx]

        # Move left
        left_idx = peak_idx - 1
        while left_idx >= 0 and Pxx[left_idx] < Pxx[left_idx + 1]:
            peak_indices.append(left_idx)
            left_idx -= 1

        # Move right
        right_idx = peak_idx + 1
        while right_idx < len(Pxx) and Pxx[right_idx] < Pxx[right_idx - 1]:
            peak_indices.append(right_idx)
            right_idx += 1

        # Add all indices in this peak to the masked set
        masked_indices.update(peak_indices)

    return peaks

def save_chunk_to_h5(processed_data, output_file, chunk_num):
    output_file_name = f"{os.path.splitext(output_file)[0]}_{chunk_num}.h5"
    with h5py.File(output_file_name, 'w') as hf:
        for i, system in enumerate(processed_data):
            group = hf.create_group(f'system_{i}')
            group.attrs['name'] = system['name']

            # Save observation duration in the main group and original_system group
            group.attrs["observation_duration"] = system["observation_duration"]

            # Save periodogram data
            group.create_dataset('periodogram_freq', data=system['periodogram_f'])
            group.create_dataset('periodogram_power', data=system['periodogram_Pxx'])

            # Save planet data
            planet_group = group.create_group('planet_data')
            for j, planet in enumerate(system['planet_data']):
                peak_group = planet_group.create_group(f'peak_{j}')
                for key, value in planet.items():
                    peak_group.attrs[key] = value

            # Save non-planet data
            non_planet_group = group.create_group('non_planet_data')
            for j, non_planet in enumerate(system['non_planet_data']):
                peak_group = non_planet_group.create_group(f'peak_{j}')
                for key, value in non_planet.items():
                    peak_group.attrs[key] = value

            # Save original system data
            orig_system = system['original_system']
            orig_group = group.create_group('original_system')
            orig_group.attrs['star_mass'] = orig_system['star_mass']
            orig_group.attrs['observation_duration'] = system["observation_duration"]  # Add this line

            planets_group = orig_group.create_group('planets')
            for j, planet in enumerate(orig_system['planets']):
                planet_group = planets_group.create_group(f'planet_{j}')
                for key, value in planet.items():
                    planet_group.attrs[key] = value

    return output_file_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process planetary system data and generate periodograms.")
    parser.add_argument("input_directory", type=str, help="Path to the directory containing JSON files of planetary systems")
    parser.add_argument("output_file", type=str, help="Path and filename for the output H5 file")
    parser.add_argument("--time_span", type=int, default=0, help="Time span in days for the simulation if left to default, will generated based on real data")
    parser.add_argument("--chunk_size", type=int, default=10000, help="Number of systems to process in each chunk (default: 10000)")
    parser.add_argument("--entropy", type=float, default=1, help="Between 0 and 1. 1 is pure chaos. 0 is purely balanced observation. (default: 1)")
    parser.add_argument("--max_files", type=int, default=20000, help="Number of files to process int h5 (default: 20000)")

    args = parser.parse_args()

    directory_path = Path(args.input_directory)
    output_file = args.output_file
    time_span = args.time_span
    chunk_size = args.chunk_size
    observation_entropy = args.entropy

    total_systems = 0
    total_size = 0

    # Load the histogram data
    bins, counts = load_histogram_data('./data/histogram_data.csv')

    # Create the time sampler
    time_sampler = create_time_sampler(bins, counts)

    for chunk_num, systems_chunk in enumerate(load_system_data_in_chunks(directory_path, chunk_size, args.max_files), 1):
        processed_data = process_systems_chunk(systems_chunk, time_span, observation_entropy, time_sampler)
        chunk_file = save_chunk_to_h5(processed_data, output_file, chunk_num)

        chunk_size = os.path.getsize(chunk_file) / (1024 * 1024)  # Size in MB
        total_size += chunk_size
        total_systems += len(processed_data)

        print(f"Processed chunk {chunk_num}: {len(processed_data)} systems")
        print(f"Chunk file: {chunk_file}")
        print(f"Chunk size: {chunk_size:.2f} MB")
        print(f"Total systems processed: {total_systems}")
        print(f"Total size: {total_size:.2f} MB")
        print("---")

    print(f"Finished processing all systems.")
    print(f"Total systems processed: {total_systems}")
    print(f"Total size of all chunks: {total_size:.2f} MB")
