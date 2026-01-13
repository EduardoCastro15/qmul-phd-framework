import pandas as pd
import numpy as np
import scipy.io as sio
import os

# Load the CSV file containing the list of food webs
csv_path = '/Users/jorge/Desktop/PhD/Code/ExtractFoodWebs/foodweb_metrics.csv'  # Replace with your actual file path
csv_data = pd.read_csv(csv_path)

# Directory paths
input_dir = '/Users/jorge/Desktop/PhD/Code/ExtractFoodWebs/foodwebs_csv/'  # Replace with the directory containing input CSVs
output_dir = '/Users/jorge/Desktop/PhD/Code/ExtractFoodWebs/foodwebs_mat_classification/'  # Replace with the directory to save .mat files
os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist

# Iterate through each food web dataset
for _, row in csv_data.iterrows():
    foodweb_name = row['Foodweb']  # Adjust column name to match your CSV
    input_file = os.path.join(input_dir, f"{foodweb_name}.csv")  # Assuming input CSV file names match the foodweb names
    
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        continue

    # Load the individual food web CSV
    web_data = pd.read_csv(input_file)

    # Step 1: Extract unique species
    consumer_species = web_data['con.taxonomy'].dropna().unique()
    resource_species = web_data['res.taxonomy'].dropna().unique()

    # Combine all species and assign a unique index to each
    all_species = pd.unique(np.concatenate((consumer_species, resource_species)))
    species_to_index = {species: idx for idx, species in enumerate(all_species)}

    # Step 2: Initialize adjacency matrix
    num_species = len(all_species)
    adjacency_matrix = np.zeros((num_species, num_species), dtype=np.double)

    # Step 3: Populate adjacency matrix based on interactions
    for _, interaction in web_data.iterrows():
        consumer = interaction['con.taxonomy']
        resource = interaction['res.taxonomy']
        
        if pd.notna(consumer) and pd.notna(resource):
            consumer_idx = species_to_index[consumer]
            resource_idx = species_to_index[resource]
            adjacency_matrix[consumer_idx, resource_idx] = 1.0  # Directed edge from consumer to resource

    # Step 4: Classify species
    species_classification = []
    for species in all_species:
        is_consumer = species in consumer_species
        is_resource = species in resource_species
        if is_consumer and is_resource:
            species_classification.append('consumer-resource')
        elif is_consumer:
            species_classification.append('consumer')
        elif is_resource:
            species_classification.append('resource')

    # Step 5: Save the new .mat file
    output_data = {
        'net': adjacency_matrix,
        'species': np.array(all_species, dtype=object),
        'classification': np.array(species_classification, dtype=object)
    }
    output_file = os.path.join(output_dir, f"{foodweb_name}.mat")
    sio.savemat(output_file, output_data)

    print(f"Processed .mat file saved to {output_file}")
