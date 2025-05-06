import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Example data: Rows are prey, Columns are predators
data = {
    'Predator1': ['TP', 'FP', 'FN', 'TN'],
    'Predator2': ['FP', 'TP', 'TN', 'FN'],
    'Predator3': ['FN', 'FN', 'TP', 'TP'],
    'Predator4': ['TN', 'TP', 'FP', 'FP'],
}
prey = ['Prey1', 'Prey2', 'Prey3', 'Prey4']

# Create a DataFrame
df = pd.DataFrame(data, index=prey)

# Map categories to colors or numerical values
color_map = {'TP': 1, 'FP': 0.5, 'FN': -0.5, 'TN': -1}
numerical_data = df.replace(color_map)

# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    numerical_data.astype(float),  # Ensure numerical data for plotting
    annot=df,  # Use the original DataFrame with strings for annotation
    fmt='',  # Ensure annotations are strings, not formatted numbers
    cmap='coolwarm',
    cbar_kws={'label': 'Prediction Quality'}
)

# Add labels
plt.title('Predation Matrix with Prediction Quality', fontsize=16)
plt.xlabel('Predators', fontsize=12)
plt.ylabel('Prey', fontsize=12)
plt.tight_layout()

# Show plot
plt.show()
