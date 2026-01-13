import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Load the CSV list of all food web files
foodweb_list_path = '/Users/jorge/Desktop/PhD/Code/MATLAB_PARFOR/data/foodwebs_mat/foodweb_metrics.csv'
foodweb_list = pd.read_csv(foodweb_list_path)

# Extract the filenames from the CSV and construct the full file paths
base_path = '/Users/jorge/Desktop/PhD/Code/MATLAB_PARFOR/data/result/'
file_paths = [base_path + filename + '.txt' for filename in foodweb_list['Foodweb'].tolist()]

# Function to parse a file and extract subgraph size and AUC data
def parse_file(filename):
    subgraph_data = {}
    try:
        with open(filename, 'r') as file:
            for line in file:
                match = re.search(r'(\d+)\s*\|\s*([\d.]+)\s*\|\s*[\d:]+\s*\|\s*(\d+)\s*\|\s*([\d%]+)', line)
                if match:
                    iteration = int(match.group(1))
                    auc = float(match.group(2))
                    subgraph_size = int(match.group(3))
                    if subgraph_size not in subgraph_data:
                        subgraph_data[subgraph_size] = []
                    subgraph_data[subgraph_size].append(auc)
    except FileNotFoundError:
        print(f"File {filename} not found.")
    return subgraph_data

# Function to calculate average AUC per subgraph size K across multiple files
def average_auc_per_k(files):
    combined_data = {}
    for file in files:
        data = parse_file(file)
        for k, aucs in data.items():
            if k not in combined_data:
                combined_data[k] = []
            combined_data[k].extend(aucs)
    
    avg_auc_per_k = {k: np.mean(aucs) for k, aucs in combined_data.items() if len(aucs) > 0}
    return avg_auc_per_k

# Calculate average AUC for each K across all files in the list
avg_auc = average_auc_per_k(file_paths)

# Plot the customized histogram
ks = list(avg_auc.keys())
avg_aucs = list(avg_auc.values())
cmap = cm.get_cmap('viridis', len(ks))

plt.figure(figsize=(18, 8))
bars = plt.bar(ks, avg_aucs, color=cmap(np.linspace(0, 1, len(ks))))

# Add value annotations on top of each bar
for bar, auc in zip(bars, avg_aucs):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{auc:.2f}', ha='center', va='bottom', fontsize=13)

# Customize x and y axis labels and title
plt.xlabel('Encoded Subgraph Size (K)', fontsize=14)
plt.ylabel('Average AUC', fontsize=14)
plt.title('Average AUC for Each Encoded Subgraph Size (K) Across Food Webs', fontsize=16)

# Rotate x-axis labels for better readability
plt.xticks(ks, rotation=45, ha='right')

# Add gridlines and customize style
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Show the plot
plt.tight_layout()
plt.show()
