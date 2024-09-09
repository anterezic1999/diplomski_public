import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('planet_predictions.csv')

# Define the threshold range
thresholds = np.arange(0.5, 0.991, 0.001)

# Initialize arrays to store counts
tn = []
fp = []
fn = []
tp = []

# Calculate counts for each threshold
for threshold in thresholds:
    tn.append(((df['Prediction'] < threshold) & (df['Is True Planet'] == 0)).sum())
    fp.append(((df['Prediction'] >= threshold) & (df['Is True Planet'] == 0)).sum())
    fn.append(((df['Prediction'] < threshold) & (df['Is True Planet'] == 1)).sum())
    tp.append(((df['Prediction'] >= threshold) & (df['Is True Planet'] == 1)).sum())

# Create the plot
plt.figure(figsize=(12, 8))
plt.plot(thresholds, tp, label='True Positives')
plt.plot(thresholds, tn, label='True Negatives')
plt.plot(thresholds, fp, label='False Positives')
plt.plot(thresholds, fn, label='False Negatives')

# Customize the plot
plt.xlabel('Detection Threshold')
plt.ylabel('Count')
plt.title('Confusion Matrix Counts vs Detection Threshold')
plt.legend()
plt.grid(True)

# Set x-axis ticks
plt.xticks(np.arange(0.5, 1.0, 0.05))

# Show the plot
plt.tight_layout()
plt.show()
