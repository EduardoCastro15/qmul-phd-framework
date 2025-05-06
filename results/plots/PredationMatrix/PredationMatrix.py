import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Example data for a predation matrix
# Rows represent prey and columns represent predators
data = {
    'cephalopod': [42, 25, 0, 0, 0, 2, 0, 0, 1, 1, 0, 5],
    'fish': [230, 5, 4, 3, 0, 2, 0, 0, 1, 2, 1, 1],
    'crustacean': [5, 46, 1, 1, 1, 2, 1, 0, 5, 11, 3, 2],
    'chaetognath': [0, 1, 0, 3, 0, 0, 0, 0, 1, 1, 0, 1],
    'polychaete': [0, 0, 0, 1, 9, 2, 0, 0, 0, 2, 0, 0],
    'protist': [0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0],
    'ctenophore': [0, 0, 0, 0, 0, 0, 11, 1, 2, 0, 0, 0],
    'scyphozoan': [0, 0, 0, 0, 0, 0, 0, 2, 5, 0, 0, 0],
    'narcomedusa': [0, 0, 0, 0, 0, 0, 0, 2, 11, 0, 0, 0],
    'hydromedusa': [3, 0, 0, 0, 0, 0, 0, 0, 9, 6, 1, 0],
    'siphonophore': [2, 0, 0, 0, 0, 0, 0, 0, 7, 4, 21, 3],
    'copepod': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2]
}

# Row labels (prey types)
prey_types = [
    'cephalopod', 'fish', 'crustacean', 'chaetognath', 'polychaete', 
    'protist', 'ctenophore', 'scyphozoan', 'narcomedusa', 
    'hydromedusa', 'siphonophore', 'copepod'
]

# Create a DataFrame
df = pd.DataFrame(data, index=prey_types)

# Log-transform the data to enhance color discrimination for low values
log_transformed = np.log10(df + 1)  # Add 1 to avoid log(0)

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(
    log_transformed, 
    annot=df, 
    fmt='g', 
    cmap='viridis', 
    cbar_kws={'label': 'log(#)'},
    linewidths=0.5,
    linecolor='black'
)

# Labels and title
plt.title("Predation Matrix", fontsize=16)
plt.xlabel("Predator", fontsize=12)
plt.ylabel("Prey", fontsize=12)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
