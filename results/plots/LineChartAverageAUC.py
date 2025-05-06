import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def calculate_average_auc_for_k_range(filenames, k_values, foodweb_names):
    # Define a regular expression pattern to match each row in the file
    pattern = r"\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*\d{2}:\d{2}:\d{2}\s*\|\s*(\d+)\s*\|\s*\d+%"
    
    # Initialize a dictionary to store the average AUC scores for each food web across k values
    auc_scores = {k: {} for k in k_values}

    # Iterate through each file and calculate the average AUC for each k value
    for filename, foodweb_name in zip(filenames, foodweb_names):
        try:
            # Read each file and calculate the average AUC for each specified k
            with open(filename, 'r') as file:
                lines = file.readlines()

            for k in k_values:
                all_aucs = [float(re.match(pattern, line).group(2))
                            for line in lines
                            if re.match(pattern, line) and int(re.match(pattern, line).group(3)) == k]

                # Store the average AUC score for this k and food web
                auc_scores[k][foodweb_name] = np.mean(all_aucs) if all_aucs else None
        except FileNotFoundError:
            print(f"File not found: {filename}")
            for k in k_values:
                auc_scores[k][foodweb_name] = None  # Handle missing data

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

# Define the range of K values from 5 to 15
# k_values = range(5, 16)
k_values = (5, 10, 15)

# Calculate the average AUC for each food web and each k value
auc_scores = calculate_average_auc_for_k_range(file_paths, k_values, foodweb_names)

# Truncate food web names to a maximum of 15 characters for plotting
truncated_foodwebs = [fw[:15] for fw in foodweb_names]
x = np.arange(len(truncated_foodwebs))  # X-axis positions for each food web

# Choose a color map and generate colors for each k value
cmap = cm.get_cmap('viridis', len(k_values))  # 'viridis' can be replaced with any other colormap

# Plot each k value with colors from the colormap
plt.figure(figsize=(19, 9))
for idx, k in enumerate(k_values):
    auc_values = [auc_scores[k][fw] for fw in foodweb_names if auc_scores[k][fw] is not None]
    plt.plot(x[:len(auc_values)], auc_values, marker='o', linestyle='None', linewidth=1.5, 
             label=f'K={k}', color=cmap(idx))  # Use cmap(idx) to get a color for each k


# Customize x-axis with truncated food web names and display every nth label to avoid clutter
plt.xticks(x[::10], [truncated_foodwebs[i] for i in range(0, len(truncated_foodwebs), 10)], rotation=45, fontsize=8, ha='right')

# Labels and title
plt.xlabel('Food Webs', fontsize=14)
plt.ylabel('Average AUC Score', fontsize=14)
plt.title('Average AUC Score for Each Food Web Across K Values', fontsize=16)

# Add legend
plt.legend(title='Enclosed subrgaph size k', loc='lower right', fontsize=15)

# Grid and layout adjustments
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()

# Show the plot
plt.show()
