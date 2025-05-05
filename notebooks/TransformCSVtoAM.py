import pandas as pd
import numpy as np

# Load the food web CSV file
foodweb_csv_path = 'foodwebs/Weddell Sea.csv'  # Replace with the path to the foodweb CSV file
foodweb_data = pd.read_csv(foodweb_csv_path)

# Extract the relevant columns: con.taxonomy (consumer) and res.taxonomy (resource)
consumers = foodweb_data['con.taxonomy']
resources = foodweb_data['res.taxonomy']

# Combine consumers and resources into a unique list of species (nodes)
species = pd.concat([consumers, resources]).unique()

# Create a mapping of species to indices
species_to_index = {species: index for index, species in enumerate(species)}

# Initialize an adjacency matrix of size (n_species, n_species), where n_species is the number of unique species
n_species = len(species)
adj_matrix = np.zeros((n_species, n_species))

# Fill the adjacency matrix
for consumer, resource in zip(consumers, resources):
    # Get the indices of the consumer and resource
    consumer_idx = species_to_index[consumer]
    resource_idx = species_to_index[resource]
    
    # Set the matrix entry to 1 to indicate a link (consumer-resource interaction)
    adj_matrix[consumer_idx, resource_idx] = 1

# Convert the adjacency matrix to a DataFrame for easier inspection
adjacency_df = pd.DataFrame(adj_matrix, index=species, columns=species)

# Optionally, save the adjacency matrix to a CSV file
adjacency_df.to_csv('foodweb_adjacency_matrix.csv')

# Print the adjacency matrix
print(adjacency_df)
