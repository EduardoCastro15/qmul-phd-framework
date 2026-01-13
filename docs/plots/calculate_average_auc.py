import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps

def calculate_average_auc(filenames, k, foodweb_names):
    # Define a regular expression pattern to match each row in the file
    pattern = r"\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*\d{2}:\d{2}:\d{2}\s*\|\s*(\d+)\s*\|\s*\d+%"
    
    # Initialize a dictionary to store the average AUC score for each food web
    auc_scores = {}

    # Iterate through each file and its corresponding food web name
    for filename, foodweb_name in zip(filenames, foodweb_names):
        all_aucs = []
        try:
            # Open and read the file
            with open(filename, 'r') as file:
                for line in file:
                    # Check if the line matches the data row pattern
                    match = re.match(pattern, line)
                    if match:
                        # Extract iteration, AUC, and encoded subgraph values
                        iteration = int(match.group(1))
                        auc = float(match.group(2))
                        subgraph_k = int(match.group(3))
                        
                        # Append to all_aucs if subgraph matches the specified K
                        if subgraph_k == k:
                            all_aucs.append(auc)
            
            # Calculate the average AUC for the specified subgraph K and store it in the dictionary
            if all_aucs:
                average_auc = sum(all_aucs) / len(all_aucs)
                auc_scores[foodweb_name] = average_auc
            else:
                auc_scores[foodweb_name] = None  # or some placeholder if no data for K

        except FileNotFoundError:
            print(f"File not found: {filename}")
            auc_scores[foodweb_name] = None  # No data if file is missing

    return auc_scores


# Load the CSV list of all food web files
foodweb_list_path = '/Users/jorge/Desktop/PhD/Code/MATLAB_PARFOR/data/foodwebs_mat/foodweb_metrics.csv'
foodweb_list = pd.read_csv(foodweb_list_path)

# Sort the data in ascending order based on the 'Edges' column
foodweb_data_sorted = foodweb_list.sort_values(by='Edges', ascending=True)

# Extract the filenames and names from the sorted data
base_path = '/Users/jorge/Desktop/PhD/Code/MATLAB_PARFOR/data/result/'
file_paths = [base_path + filename + '.txt' for filename in foodweb_data_sorted['Foodweb'].tolist()]
foodweb_names = foodweb_data_sorted['Foodweb'].tolist()

# Set the value of K
k = 15

# Calculate the average AUC for each food web and store in a dictionary
auc_scores = calculate_average_auc(file_paths, k, foodweb_names)

# Filter out None values for plotting
filtered_foodwebs = [fw for fw, score in auc_scores.items() if score is not None]
filtered_auc_scores = [score for score in auc_scores.values() if score is not None]

# Truncate food web names to a maximum of 15 characters
truncated_foodwebs = [fw[:15] for fw in filtered_foodwebs]

# Plot the bar chart for K with the food webs on the x-axis
plt.figure(figsize=(19, 9))

x = np.arange(len(truncated_foodwebs))  # X-axis positions for each food web
norm = plt.Normalize(vmin=min(filtered_auc_scores), vmax=max(filtered_auc_scores))
cmap = colormaps.get_cmap('Reds')

# Map AUC values to colors directly
colors = cmap(norm(filtered_auc_scores))
bars = plt.bar(x, filtered_auc_scores, color=colors, width=0.6)

# Add data labels for each bar
for bar, auc in zip(bars, filtered_auc_scores):
    if not np.isnan(auc):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{auc:.2f}', ha='center', va='bottom', fontsize=7)
    else:
        plt.text(bar.get_x() + bar.get_width() / 2, 0.02, 'NaN', ha='center', va='bottom', fontsize=7, color='red')

# Customize x-axis with food web names and display every nth label to avoid clutter
plt.xticks(x[::10], [truncated_foodwebs[i] for i in range(0, len(truncated_foodwebs), 10)], rotation=45, fontsize=8, ha='right')

# Labels and title
plt.xlabel('Food Webs', fontsize=14)
plt.ylabel(f'Average AUC for K={k}', fontsize=14)
plt.title(f'Average AUC for K={k} for Each Food Web (Sorted by Ascending Edge Count)', fontsize=16)

# Adjust spacing on x-axis for padding
plt.xlim(-1, len(truncated_foodwebs))  # Add padding to the left and right of the bars

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
