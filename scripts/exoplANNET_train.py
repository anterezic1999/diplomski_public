#TRAINING:
import os
import h5py
import argparse
import matplotlib.pyplot as plt
from torchview import draw_graph
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm
import random
import numpy as np
import torch.nn.functional as F

PERIODOGRAM_LEN = 1000
SOLAR_MASS = 1.9885e30

def check_nan(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
        return True
    return False

def load_h5_files(directory):
    data = []
    h5_files = [f for f in os.listdir(directory) if f.endswith('.h5')]

    for filename in tqdm(h5_files, desc="Loading H5 files", unit="file"):
        file_path = os.path.join(directory, filename)
        with h5py.File(file_path, 'r') as hf:
            for system_name in hf.keys():
                system = hf[system_name]
                system_data = {
                    'name': system.attrs['name'],
                    'periodogram_f': np.array(system['periodogram_freq']),
                    'periodogram_Pxx': np.array(system['periodogram_power']),
                    'planet_data': [],
                    'non_planet_data': []
                }

                # Access star_mass from original_system subgroup
                if 'original_system' in system:
                    original_system = system['original_system']
                    if 'star_mass' in original_system.attrs:
                        system_data['star_mass_solar_mass'] = original_system.attrs['star_mass'] / SOLAR_MASS
                    else:
                        print(f"Warning: 'star_mass' not found in original_system attributes for {system_name}")
                else:
                    print(f"Warning: 'original_system' group not found for {system_name}")

                # Load planet data
                if 'planet_data' in system:
                    for peak in system['planet_data'].values():
                        planet = {key: value for key, value in peak.attrs.items()}
                        system_data['planet_data'].append(planet)

                # Load non-planet data
                if 'non_planet_data' in system:
                    for peak in system['non_planet_data'].values():
                        non_planet = {key: value for key, value in peak.attrs.items()}
                        system_data['non_planet_data'].append(non_planet)

                data.append(system_data)

    return data

def generate_list_of_possible_planets(system, freq_to_avoid):
    arrayOfFrequencies = []
    countOfPeaks = random.randint(0, 4)

    planet_frequencies = [planet['peak_frequency'] for planet in system['planet_data'] if abs(planet['peak_frequency'] - freq_to_avoid) > 1e-6]
    non_planet_frequencies = [non_planet['peak_frequency'] for non_planet in system['non_planet_data'] if abs(non_planet['peak_frequency'] - freq_to_avoid) > 1e-6]

    used_planet_indices = set()
    used_non_planet_indices = set()

    for i in range(countOfPeaks):
        if np.random.uniform(0.0, 1.0) < 0.8747:
            # Select an actual planet frequency
            available_planet_indices = set(range(len(planet_frequencies))) - used_planet_indices
            if available_planet_indices:
                index = random.choice(list(available_planet_indices))
                frequency = planet_frequencies[index]
                arrayOfFrequencies.append(frequency)
                used_planet_indices.add(index)
            else:
                arrayOfFrequencies.append(0)
        else:
            # Select a random nearby peak frequency (non-planet)
            available_non_planet_indices = set(range(len(non_planet_frequencies))) - used_non_planet_indices
            if available_non_planet_indices:
                index = random.choice(list(available_non_planet_indices))
                frequency = non_planet_frequencies[index]
                arrayOfFrequencies.append(frequency)
                used_non_planet_indices.add(index)
            else:
                arrayOfFrequencies.append(0)

    # Ensure the array has exactly 4 elements and all are floats
    arrayOfFrequencies = [float(f) for f in arrayOfFrequencies[:4]]
    while len(arrayOfFrequencies) < 4:
        arrayOfFrequencies.append(0.0)

    # Randomly shuffle the frequencies
    random.shuffle(arrayOfFrequencies)

    return arrayOfFrequencies

def prepare_training_data(data):
    X = []
    y = []

    total_items = sum(len(system['planet_data']) + len(system['non_planet_data']) for system in data)

    with tqdm(total=total_items, desc="Preparing data") as pbar:
        for system in data:
            periodogram_f = system['periodogram_f']
            periodogram_Pxx = system['periodogram_Pxx']
            star_mass = system.get('star_mass_solar_mass', 0)  # Default to 0 if not available

            for planet in system['planet_data']:
                row = np.concatenate([
                    periodogram_f,  # All frequency values
                    periodogram_Pxx,  # All power values
                    [planet['peak_frequency'], planet['peak_power'], star_mass],
                    planet['focused_peak']
                ])
                X.append(row)
                y.append(1)  # 1 for planet
                pbar.update(1)

            for non_planet in system['non_planet_data']:
                row = np.concatenate([
                    periodogram_f,
                    periodogram_Pxx,
                    [non_planet['peak_frequency'], non_planet['peak_power'], star_mass],
                    non_planet['focused_peak']
                ])
                X.append(row)
                y.append(0)  # 0 for non-planet
                pbar.update(1)

    X = np.array(X)
    y = np.array(y)

    initial_rows = X.shape[0]

    # Final check for NaN values
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    y = y[mask]

    final_rows = X.shape[0]
    dropped_rows = initial_rows - final_rows

    if dropped_rows > 0:
        print(f"Dropped {dropped_rows} rows due to NaN values")

    return X, y

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

def normalize_data(X):
    # Log1p transform frequency and power
    freq_log = np.log1p(X[:, :PERIODOGRAM_LEN])
    #power_log = np.log1p(X[:, PERIODOGRAM_LEN:PERIODOGRAM_LEN*2])
    power_log = X[:, PERIODOGRAM_LEN:PERIODOGRAM_LEN*2]

    peak_freq_log = np.log1p(X[:, PERIODOGRAM_LEN*2:PERIODOGRAM_LEN*2+1])
    peak_power_log = np.log1p(X[:, PERIODOGRAM_LEN*2+1:PERIODOGRAM_LEN*2+2])

    #star_mass = X[:, PERIODOGRAM_LEN*2+2:PERIODOGRAM_LEN*2+3]  # Keep as is
    star_mass = np.zeros((X.shape[0], 1))

    #additional_freq = np.log1p(X[:, PERIODOGRAM_LEN*2+3:])
    #shuffled_additional_freq = np.array([np.random.permutation(row) for row in additional_freq])

    normalized_data = np.column_stack((
        freq_log,
        power_log,
        peak_freq_log,
        peak_power_log,
        star_mass,
        X[:, PERIODOGRAM_LEN*2+3:] #This is the selected peak
        #shuffled_additional_freq
    ))

    return normalized_data

def plot_periodogram(f, Pxx):
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(f, Pxx)
    plt.xlabel('Period (days)')
    plt.ylabel('Power')
    plt.title(f'Periodogram')
    plt.xscale('log')

    # Reverse x-axis so shorter periods are on the left
    plt.gca().invert_xaxis()

    # Set x-axis limits to show relevant period range
    plt.xlim(max(f), min(f))

    # Format y-axis to use scientific notation
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

    plt.tight_layout()
    plt.show()

    plt.close()

def plot_training_history(train_losses, val_losses, f1_scores, epoch):
    plt.figure(figsize=(16, 10), dpi=300)  # Increased figure size and DPI for higher resolution

    epochs = range(1, len(train_losses) + 1)

    # Plot all metrics on a single plot
    plt.plot(epochs, train_losses, label='Training Loss', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='s')
    plt.plot(epochs, f1_scores, label='F1 Score', marker='^')

    plt.title('Training History', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Metrics', fontsize=14)
    plt.legend(fontsize=12)

    # Set integer ticks for x-axis
    plt.xticks(epochs)

    # Increase font size of tick labels
    plt.tick_params(axis='both', which='major', labelsize=12)

    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()

    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.close()

def train_model(X_normalized, y_balanced, batch_size=128, epochs=250, learning_rate=1e-4, patience=15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize lists to store history
    train_losses = []
    val_losses = []
    f1_scores = []

    X_train, X_val, y_train, y_val = train_test_split(X_normalized, y_balanced, test_size=0.2, random_state=42)
    print(X_train.shape)

    # Correctly separate periodogram frequencies, powers, and features
    X_train_freq = torch.FloatTensor(X_train[:, :PERIODOGRAM_LEN])
    X_train_power = torch.FloatTensor(X_train[:, PERIODOGRAM_LEN:PERIODOGRAM_LEN*2])
    X_train_features = torch.FloatTensor(X_train[:, PERIODOGRAM_LEN*2:PERIODOGRAM_LEN*2+3])
    X_train_peak = torch.FloatTensor(X_train[:, PERIODOGRAM_LEN*2+3:])
    y_train = torch.FloatTensor(y_train)


    X_val_freq = torch.FloatTensor(X_val[:, :PERIODOGRAM_LEN])
    X_val_power = torch.FloatTensor(X_val[:, PERIODOGRAM_LEN:PERIODOGRAM_LEN*2])
    X_val_features = torch.FloatTensor(X_val[:, PERIODOGRAM_LEN*2:PERIODOGRAM_LEN*2+3])
    X_val_peak = torch.FloatTensor(X_val[:, PERIODOGRAM_LEN*2+3:])
    y_val = torch.FloatTensor(y_val)

    train_dataset = TensorDataset(X_train_freq, X_train_power, X_train_features, X_train_peak, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    val_dataset = TensorDataset(X_val_freq, X_val_power, X_val_features, X_val_peak, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)

    model = PlanetDetectionModel_Enhanced().to(device)

    # Create sample inputs
    batch_size = 1
    freq = torch.rand(batch_size, 128).to(device)
    power = torch.rand(batch_size, 128).to(device)
    features = torch.rand(batch_size, 3).to(device)
    peak = torch.rand(batch_size, 128).to(device)

    # Generate the model graph
    model_graph = draw_graph(
        model,
        input_data=(freq, power, features, peak),
        expand_nested=True,
        hide_inner_tensors=True,
        hide_module_functions=True
    )

    # Generate the current date and time in the format mm_dd_hh_mm
    formatted_date = datetime.now().strftime('%m_%d_%H_%M')

    # Use the formatted date in the filename
    model_graph.visual_graph.render(f"planet_detection_model_{formatted_date}", format="png")

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)

    scaler = GradScaler()

    best_f1_score = 0
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
        for freq, power, features, peak, labels in batch_pbar:
            freq, power, features, peak, labels = freq.to(device), power.to(device), features.to(device), peak.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(freq, power, features, peak)
                loss = criterion(outputs, labels.unsqueeze(1))
            scaler.scale(loss).backward()
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            batch_pbar.set_postfix({'batch_loss': f'{loss.item():.4f}'})

        # Similar changes for the validation loop
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_true = []
        with torch.no_grad():
            for freq, power, features, peak, labels in val_loader:
                freq, power, features, peak, labels = freq.to(device), power.to(device), features.to(device), peak.to(device), labels.to(device)
                with autocast():
                    outputs = model(freq, power, features, peak)
                    loss = criterion(outputs, labels.unsqueeze(1))
                val_loss += loss.item()
                val_preds.extend(outputs.squeeze().cpu().numpy())
                val_true.extend(labels.cpu().numpy())

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        val_preds_binary = (torch.sigmoid(torch.tensor(val_preds)) > 0.5).numpy().astype(int)
        f1 = f1_score(val_true, val_preds_binary)

        # Append to history lists
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        f1_scores.append(f1)

        # Plot and save training history
        plot_training_history(train_losses, val_losses, f1_scores, epoch)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, F1-score: {f1:.4f}")

        scheduler.step(f1)

        if f1 > best_f1_score:
            best_f1_score = f1
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'New best model saved with F1-score: {f1:.4f}')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f'Early stopping triggered. Best F1-score: {best_f1_score:.4f}')
                break

    print(f'Training completed. Best F1-score: {best_f1_score:.4f}')
    return model
def main():
    parser = argparse.ArgumentParser(description="Load, prepare, and balance data from H5 files.")
    parser.add_argument("path", help="Path to the directory containing H5 files")
    args = parser.parse_args()

    data = load_h5_files(args.path)

    X, y = prepare_training_data(data)

    print("Final dataset:")
    print(f"Total samples: {len(y)}")
    print(f"Ratio of planet data: {sum(y)/len(y)}")
    print(f"Feature vector shape: {X[0].shape}")

    print("Normalizing data...")
    X_normalized = normalize_data(X)

    print("Training Model:")
    model = train_model(X_normalized, y)

    torch.save(model, 'planet_detection_model_full.pth')
    print("Full model saved as planet_detection_model_full.pth")

if __name__ == "__main__":
    main()
