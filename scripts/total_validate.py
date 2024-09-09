import os
import h5py
import argparse
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.cuda.amp import autocast
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import torch.nn.functional as F

PERIODOGRAM_LEN = 1000
DETECTION_THRESHOLD = 0.5
SOLAR_MASS = 1.9885e30

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if out.size() != identity.size():
            identity = F.pad(identity, (0, out.size(2) - identity.size(2)))

        out += self.shortcut(identity)
        out = F.relu(out)
        return out

class AttentionLayer(nn.Module):
    def __init__(self, channels):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(channels // 8, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights

class PlanetDetectionModel_Enhanced(nn.Module):
    def __init__(self):
        super(PlanetDetectionModel_Enhanced, self).__init__()

        self.conv_layers = nn.Sequential(
            ResidualBlock(2, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),
            #AttentionLayer(32),
            ResidualBlock(32, 64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),
            #AttentionLayer(64),
            ResidualBlock(64, 128, kernel_size=7, stride=1, padding=3),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),
            #AttentionLayer(128),
            ResidualBlock(128, 256, kernel_size=9, stride=1, padding=4),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),
            #AttentionLayer(256),
            ResidualBlock(256, 512, kernel_size=11, stride=1, padding=5),
            #AttentionLayer(512),
            nn.AdaptiveAvgPool1d(1)
        )

        self.peak_conv_layers = nn.Sequential(
            ResidualBlock(1, 16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),
            #AttentionLayer(16),
            ResidualBlock(16, 32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),
            #AttentionLayer(32),
            ResidualBlock(32, 64, kernel_size=7, stride=1, padding=3),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),
            #AttentionLayer(64),
            nn.AdaptiveAvgPool1d(1)
        )

        self.fc_features = nn.Sequential(
            nn.Linear(3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.fc_combined = nn.Sequential(
            nn.Linear(512 + 64 + 64, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 6),
            nn.BatchNorm1d(6),
            nn.ReLU(),
            nn.Linear(6, 1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, freq, power, features, peak):
        periodogram = torch.stack((freq, power), dim=1)
        conv_out = self.conv_layers(periodogram).squeeze(-1)
        peak_conv_out = self.peak_conv_layers(peak.unsqueeze(1)).squeeze(-1)
        features_out = self.fc_features(features)
        combined = torch.cat((conv_out, peak_conv_out, features_out), dim=1)
        output = self.fc_combined(combined)
        return output

def normalize_periodogram(data):
    # Extract frequency and power
    freq = data[:PERIODOGRAM_LEN]
    power = data[PERIODOGRAM_LEN:PERIODOGRAM_LEN*2]

    # Apply log1p transformation
    freq_log = np.log1p(freq)
    power_log = np.log1p(power)

    # # Normalize using the provided mean and std values
    # freq_mean, freq_std = 3.2810406108236996, 4.036496125329278
    # power_mean, power_std = 3.5355131461859264, 3.3413114910405226
    #
    # freq_normalized = (freq_log - freq_mean) / freq_std
    # power_normalized = (power_log - power_mean) / power_std


    # Combine normalized data
    normalized_data = np.concatenate([
        freq_log,
        power_log
    ])

    return normalized_data

def load_h5_file(file_path):
    with h5py.File(file_path, 'r') as hf:
        systems = []
        for system_name in hf.keys():
            system = hf[system_name]
            orig_system = system['original_system']
            # Access the star mass from the original_system attributes
            star_mass = orig_system.attrs.get('star_mass', 0) / SOLAR_MASS

            # Load planet data
            planet_data = []
            if 'planet_data' in system:
                for peak_name in system['planet_data']:
                    peak = system['planet_data'][peak_name]
                    peak_data = {key: peak.attrs[key] for key in peak.attrs.keys()}
                    planet_data.append(peak_data)

            # Load non-planet data if needed
            non_planet_data = []
            if 'non_planet_data' in system:
                for peak_name in system['non_planet_data']:
                    peak = system['non_planet_data'][peak_name]
                    peak_data = {key: peak.attrs[key] for key in peak.attrs.keys()}
                    non_planet_data.append(peak_data)

            # Load original system data
            original_system = {}
            if 'original_system' in system:
                orig_system = system['original_system']
                original_system['star_mass'] = orig_system.attrs['star_mass']
                original_system['planets'] = []
                if 'planets' in orig_system:
                    for planet_name in orig_system['planets']:
                        planet = orig_system['planets'][planet_name]
                        planet_data = {key: planet.attrs[key] for key in planet.attrs.keys()}
                        original_system['planets'].append(planet_data)

            system_data = {
                'name': system.attrs['name'],
                'planet_data': planet_data,
                'star_mass': star_mass,  # Default to 0 if not available
                'non_planet_data': non_planet_data,
                'periodogram_f': np.array(system['periodogram_freq']),
                'periodogram_Pxx': np.array(system['periodogram_power']),
                'original_system': original_system
            }
            systems.append(system_data)
    return systems

def evaluate_model(model, star_mass, periodogram_norm, selected_indices, frequencies, power, selected_peaks, device):
    model.eval()
    predictions = []

    # Split the periodogram_norm into power and frequencies
    frequencies_norm = periodogram_norm[:PERIODOGRAM_LEN]
    power_norm = periodogram_norm[PERIODOGRAM_LEN:]

    # Convert to PyTorch tensors and move to device
    power_tensor = torch.FloatTensor(power_norm).unsqueeze(0).to(device)
    frequencies_tensor = torch.FloatTensor(frequencies_norm).unsqueeze(0).to(device)

    with torch.no_grad():
        for i, peak_idx in enumerate(selected_indices):
            peak_freq = frequencies[peak_idx]
            peak_power = power[peak_idx]

            # Normalize frequency and power
            freq_norm = np.log1p(peak_freq)
            pow_norm = np.log1p(peak_power)

            # Prepare the feature tensor for this specific frequency/power pair
            features = torch.FloatTensor([freq_norm, pow_norm, star_mass]).unsqueeze(0).to(device)

            # Get the corresponding peak from selected_peaks and reshape
            entire_peak = torch.FloatTensor(selected_peaks[i]).unsqueeze(0).to(device)  # Shape: [1, 61]

            with autocast():
                output = model(frequencies_tensor, power_tensor, features, entire_peak)

            prediction = torch.sigmoid(output).item()
            predictions.append(prediction)

    return predictions

def plot_planet_likelihood(system_name, frequencies, likelihoods, output_dir):
    plt.figure(figsize=(12, 6))
    plt.plot(frequencies, likelihoods)
    plt.xlabel('Frequency')
    plt.ylabel('Planet Likelihood')
    plt.title(f'Planet Likelihood for System: {system_name}')
    plt.ylim(0, 1)
    plt.savefig(os.path.join(output_dir, f'{system_name}_planet_likelihood.png'))
    plt.close()

def plot_results(system, frequencies, power, top_10_frequencies, predictions, output_dir):
    plt.figure(figsize=(12, 6))

    # Convert frequencies to periods
    periods = 1 / frequencies
    top_10_periods = 1 / top_10_frequencies

    plt.plot(periods, power, linewidth=0.7)
    plt.xlabel('Period (days)')
    plt.ylabel('Power')
    plt.title(f'Periodogram for System: {system["name"]}')
    plt.xscale('log')

    # Get ground truth planet periods
    true_planet_periods = [planet['P'] for planet in system['original_system']['planets']]

    for period, pred in zip(top_10_periods, predictions):
        closest_true_period = min(true_planet_periods, key=lambda x: abs(x-period)) if true_planet_periods else None
        is_true_planet = closest_true_period is not None and abs(period - closest_true_period) / closest_true_period < 0.1

        if is_true_planet:
            if pred > DETECTION_THRESHOLD:
                plt.axvline(x=period, color='g', linestyle='-', alpha=0.5)  # Correct planet prediction
            else:
                plt.axvline(x=period, color='b', linestyle='-', alpha=0.5)  # Missed planet
        else:
            if pred > DETECTION_THRESHOLD:
                plt.axvline(x=period, color='r', linestyle=':', alpha=0.5)  # False positive
            else:
                plt.axvline(x=period, color='g', linestyle=':', alpha=0.5)  # Correct non-planet

    # Set x-axis to increase from left to right
    plt.gca().invert_xaxis()

    # Set x-axis limits
    plt.xlim(max(periods), min(periods))

    plt.savefig(os.path.join(output_dir, f'{system["name"]}_periodogram.png'), dpi=300)
    plt.close()

def construct_periodogram(power, frequencies):
    # Ensure power and frequencies are the same length
    assert len(power) == len(frequencies) == PERIODOGRAM_LEN, "Power and frequencies must both have length PERIODOGRAM_LEN"

    # Create the new array
    periodogram = np.zeros(PERIODOGRAM_LEN*2)

    # Fill the first PERIODOGRAM_LEN elements with power values
    periodogram[:PERIODOGRAM_LEN] = frequencies

    # Fill the next PERIODOGRAM_LEN elements with frequency values
    periodogram[PERIODOGRAM_LEN:PERIODOGRAM_LEN*2] = power

    return periodogram

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

def select_top_frequencies_exclude_entire_peak(frequencies, power, n, max_period=7000, PERIODOGRAM_LEN=61):
    min_freq = 1 / max_period
    selected_indices = []
    selected_frequencies = []
    masked_indices = set()
    peaks = []

    while len(selected_indices) < n and len(masked_indices) < len(frequencies):
        # Find the highest power frequency that's not masked
        available_indices = [i for i in range(len(frequencies)) if i not in masked_indices]
        if not available_indices:
            break
        top_idx = max(available_indices, key=lambda i: power[i])

        if frequencies[top_idx] < min_freq:
            break

        peak_indices = [top_idx]
        peak_period = 1 / frequencies[top_idx]

        # Move left
        left_idx = top_idx - 1
        while left_idx >= 0 and power[left_idx] < power[left_idx + 1]:
            peak_indices.append(left_idx)
            left_idx -= 1

        # Move right
        right_idx = top_idx + 1
        while right_idx < len(frequencies) and power[right_idx] < power[right_idx - 1]:
            peak_indices.append(right_idx)
            right_idx += 1

        # Add the peak frequency to selected list
        selected_indices.append(top_idx)
        selected_frequencies.append(frequencies[top_idx])

        # Mask all indices in this peak
        masked_indices.update(peak_indices)

        # Mask all periods within 10% of the selected peak
        for i in range(len(frequencies)):
            if i not in masked_indices:
                period = 1 / frequencies[i]
                if abs(period - peak_period) / peak_period <= 0.1:
                    masked_indices.add(i)

        # Get the padded peak and add it to peaks
        padded_peak = get_peak_padded(top_idx, frequencies, power)
        padded_peak = pad_truncate(padded_peak, PERIODOGRAM_LEN)
        peaks.append(padded_peak)

        if len(selected_indices) == n:
            break

    return np.array(selected_indices), np.array(peaks)

def main():
    parser = argparse.ArgumentParser(description="Evaluate planet detection model on periodogram data.")
    parser.add_argument("model_path", help="Path to the trained model .pth file")
    parser.add_argument("data_path", help="Path to the directory containing H5 files for evaluation")
    parser.add_argument("output_dir", help="Directory to save the output plots")
    parser.add_argument("--n_plots", type=int, default=0, help="Number of systems to plot (default: 0)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model
    model = PlanetDetectionModel_Enhanced().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    peak_values = range(3, 5)
    threshold_values = np.arange(0.65, 0.85, 0.025)
    f1_scores = np.zeros((len(peak_values), len(threshold_values)))
    precision_scores = np.zeros((len(peak_values), len(threshold_values)))
    recall_scores = np.zeros((len(peak_values), len(threshold_values)))

    for i, peaks in enumerate(peak_values):
        all_true_labels = []
        all_predictions = []

        # Load and process H5 files
        h5_files = [f for f in os.listdir(args.data_path) if f.endswith('.h5')]
        for file in h5_files:
            file_path = os.path.join(args.data_path, file)
            systems = load_h5_file(file_path)

            for system in tqdm(systems, desc=f"Processing systems (peaks={peaks})"):
                power = system['periodogram_Pxx']
                frequencies = system['periodogram_f']
                star_mass = system["star_mass"]

                selected_indices, selected_peaks = select_top_frequencies_exclude_entire_peak(frequencies, power, n=peaks)

                if len(selected_indices):
                    periodogram = construct_periodogram(power, frequencies)
                    periodogram_norm = normalize_periodogram(periodogram)
                    predictions = evaluate_model(model, 0, periodogram_norm, selected_indices, frequencies, power, selected_peaks, device)

                    true_planet_periods = [planet['P'] for planet in system['original_system']['planets']]
                    true_labels = []
                    for freq in frequencies[selected_indices]:
                        period = 1 / freq
                        closest_true_period = min(true_planet_periods, key=lambda x: abs(x-period)) if true_planet_periods else None
                        is_true_planet = closest_true_period is not None and abs(period - closest_true_period) / closest_true_period < 0.1
                        true_labels.append(int(is_true_planet))

                    all_predictions.extend(predictions)
                    all_true_labels.extend(true_labels)

        for j, threshold in enumerate(threshold_values):
            binary_predictions = [1 if pred > threshold else 0 for pred in all_predictions]
            f1_scores[i, j] = f1_score(all_true_labels, binary_predictions)
            precision_scores[i, j] = precision_score(all_true_labels, binary_predictions)
            recall_scores[i, j] = recall_score(all_true_labels, binary_predictions)

    # Plot F1 score heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(f1_scores, cmap='viridis', aspect='auto', origin='lower')
    plt.colorbar(label='F1 Score')
    plt.xlabel('Detection Threshold')
    plt.ylabel('Number of Peaks')
    plt.title('F1 Score Heatmap')
    plt.xticks(range(len(threshold_values)), [f'{t:.2f}' for t in threshold_values], rotation=45)
    plt.yticks(range(len(peak_values)), peak_values)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'f1_score_heatmap.png'))
    plt.close()

    # Plot Precision heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(precision_scores, cmap='viridis', aspect='auto', origin='lower')
    plt.colorbar(label='Precision')
    plt.xlabel('Detection Threshold')
    plt.ylabel('Number of Peaks')
    plt.title('Precision Heatmap')
    plt.xticks(range(len(threshold_values)), [f'{t:.2f}' for t in threshold_values], rotation=45)
    plt.yticks(range(len(peak_values)), peak_values)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'precision_heatmap.png'))
    plt.close()

    # Plot Recall heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(recall_scores, cmap='viridis', aspect='auto', origin='lower')
    plt.colorbar(label='Recall')
    plt.xlabel('Detection Threshold')
    plt.ylabel('Number of Peaks')
    plt.title('Recall Heatmap')
    plt.xticks(range(len(threshold_values)), [f'{t:.2f}' for t in threshold_values], rotation=45)
    plt.yticks(range(len(peak_values)), peak_values)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'recall_heatmap.png'))
    plt.close()

    print("Evaluation complete. Heatmaps saved in the output directory.")

if __name__ == "__main__":
    main()
