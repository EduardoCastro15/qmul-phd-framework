import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps

# Load the CSV list of all food web files
foodweb_list_path = '/Users/jorge/Desktop/PhD/Code/MATLAB_PARFOR/data/foodwebs_mat/foodweb_metrics.csv'
foodweb_list = pd.read_csv(foodweb_list_path)

# Sort the data in ascending order based on the 'Edges' column
foodweb_data_sorted = foodweb_list.sort_values(by='Edges', ascending=True)

# Extract the filenames and names from the sorted data
base_path = '/Users/jorge/Desktop/PhD/Code/MATLAB_PARFOR/data/result/'
file_paths = [base_path + filename + '.txt' for filename in foodweb_data_sorted['Foodweb'].tolist()]
foodweb_names = foodweb_data_sorted['Foodweb'].tolist()

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

# Function to extract AUC values for K across all food webs
def auc_for_k(files, k):
    auc_values = []
    
    for file in files:
        data = parse_file(file)
        if k in data:
            # Compute the average AUC for K for this food web
            auc_values.append(np.mean(data[k]))
            print(data[k])
        else:
            # If K is missing, add NaN to keep alignment in the list
            auc_values.append(np.nan)
    
    return auc_values

# Get AUC values for K across all food webs
k=11
auc_values = auc_for_k(file_paths, k)

# Truncate food web names to a maximum of 15 characters
foodweb_names = [name[:15] + '...' if len(name) > 15 else name for name in foodweb_names]

# Plot the histogram for K with the food webs on the x-axis
plt.figure(figsize=(19, 9))

x = np.arange(len(foodweb_names))  # X-axis positions for each food web
norm = plt.Normalize(vmin=min(filter(np.isfinite, auc_values)), vmax=max(filter(np.isfinite, auc_values)))
cmap = colormaps.get_cmap('Reds')

# Map AUC values to colors directly
colors = cmap(norm(auc_values))
bars = plt.bar(x, auc_values, color=colors, width=0.6)

# Add data labels for each bar
for bar, auc in zip(bars, auc_values):
    if not np.isnan(auc):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{auc:.2f}', ha='center', va='bottom', fontsize=7)
    else:
        plt.text(bar.get_x() + bar.get_width() / 2, 0.02, 'NaN', ha='center', va='bottom', fontsize=7, color='red')

# Customize x-axis with food web names and display every nth label to avoid clutter
plt.xticks(x[::10], [foodweb_names[i] for i in range(0, len(foodweb_names), 10)], rotation=45, fontsize=8, ha='right')

# Labels and title
plt.xlabel('Food Webs', fontsize=14)
plt.ylabel(f'Average AUC for K={k}', fontsize=14)
plt.title(f'Average AUC for K={k} Across Food Webs (Sorted by Ascending Edge Count)', fontsize=16)

# Adjust spacing on x-axis for padding
plt.xlim(-1, len(foodweb_names))  # Add padding to the left and right of the bars

# Adding a color bar to indicate the gradient
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=plt.gca(), shrink=1, aspect=40, pad=0.005, fraction=0.02)
cbar.set_label('AUC Value', fontsize=12)

# Grid and layout adjustments
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()

# Show the plot
plt.show()
