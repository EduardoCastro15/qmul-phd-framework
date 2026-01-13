import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np

# File paths (user's uploaded files)
file_paths = [
    "/Users/jorge/Desktop/Grand Caricaie  marsh dominated by Cladietum marisci, mown  Clmown1_Main.txt",
    "/Users/jorge/Desktop/Grand Caricaie  marsh dominated by Cladietum marisci, mown  Clmown1_negative-links-sampling.txt",
    "/Users/jorge/Desktop/Grand Caricaie  marsh dominated by Cladietum marisci, mown  Clmown1_directed-graphs-modification.txt",
    "/Users/jorge/Desktop/Grand Caricaie  marsh dominated by Cladietum marisci, mown  Clmown1_Merge_2_mods.txt",
]

file_labels = ["Main", "Negative Links Sampling", "Directed Graphs", "Merge Directed Graphs and Negative Links Sampling"]

# Regular expression pattern to extract relevant data
pattern = r"\|\s*\d+\s*\|\s*([\d.]+)\s*\|\s*\d{2}:\d{2}:\d{2}\s*\|\s*(\d+)\s*\|"

# Initialize dictionary to store average AUC scores for each K per file
auc_data = {label: {} for label in file_labels}

# Process each file and calculate averages
for file_path, label in zip(file_paths, file_labels):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Extract AUC scores and K values
    auc_scores = {}
    for line in lines:
        match = re.search(pattern, line)
        if match:
            auc, k = float(match.group(1)), int(match.group(2))
            if k not in auc_scores:
                auc_scores[k] = []
            auc_scores[k].append(auc)
    
    # Calculate averages for each K
    for k, scores in auc_scores.items():
        auc_data[label][k] = np.mean(scores)

# Prepare data for plotting
k_values = sorted(set(k for label_data in auc_data.values() for k in label_data.keys()))

plt.figure(figsize=(12, 6))

# Plot each file's data
colors = ['blue', 'green', 'orange', 'red']
for label, color in zip(file_labels, colors):
    auc_means = [auc_data[label].get(k, np.nan) for k in k_values]
    plt.plot(k_values, auc_means, marker='o', label=label, color=color, linewidth=2)
    # plt.scatter(k_values, auc_means, label=label, color=color, s=80)  # Scatter plot for separate points

# Customize plot
plt.xlabel('Encoded Subgraph Size (K)', fontsize=12)
plt.ylabel('Average AUC Score', fontsize=12)
plt.title('Average AUC Scores Across Encoded Subgraph Sizes', fontsize=14)
plt.xticks(k_values, fontsize=10)
plt.yticks(fontsize=10)
plt.legend(title='Files', fontsize=10)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
