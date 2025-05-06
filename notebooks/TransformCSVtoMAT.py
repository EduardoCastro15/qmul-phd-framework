import pandas as pd
import scipy.io as sio
from scipy.sparse import csr_matrix
import os

# Load the list of food web files (replace with your actual path)
foodweb_list_file = 'foodweb_metrics.csv'  # Path to your list of food webs
foodweb_list = pd.read_csv(foodweb_list_file)

# Assuming 'file_path' is the column containing the paths to the food web CSV files
for index, row in foodweb_list.iterrows():
    csv_file_path_origin = 'foodwebs_csv/' + row['Foodweb'] + '.csv'
    csv_file_path_dest = 'foodwebs_mat/' + row['Foodweb'] + '.mat'

    # Load the individual food web CSV file
    csv_data = pd.read_csv(csv_file_path_origin)

    # Extract the relevant columns for consumer and resource taxonomy
    taxonomy_data = csv_data[['con.taxonomy', 'res.taxonomy']]

    # Drop rows with missing data in either column (if applicable)
    taxonomy_data = taxonomy_data.dropna()

    # Generate unique mappings for species/taxa to indices (dynamic size based on CSV content)
    unique_taxa = pd.concat([taxonomy_data['con.taxonomy'], taxonomy_data['res.taxonomy']]).unique()
    taxa_to_index = {taxon: i for i, taxon in enumerate(unique_taxa)}

    # Initialize an adjacency matrix of zeros (dynamic size based on the unique taxa)
    n_taxa = len(unique_taxa)
    adjacency_matrix = pd.DataFrame(0, index=range(n_taxa), columns=range(n_taxa))

    # Populate the adjacency matrix based on the relationships
    for _, row in taxonomy_data.iterrows():
        consumer = row['con.taxonomy']
        resource = row['res.taxonomy']
        if consumer in taxa_to_index and resource in taxa_to_index:
            consumer_idx = taxa_to_index[consumer]
            resource_idx = taxa_to_index[resource]
            adjacency_matrix.at[consumer_idx, resource_idx] = 1  # Set a link from consumer to resource

    # Convert to a sparse matrix
    sparse_matrix = csr_matrix(adjacency_matrix.values)

    # Prepare the data for saving to .mat format, ensuring 'net' is the variable name
    mat_data = {
        'net': sparse_matrix  # The sparse matrix, named 'net'
    }

    # Ensure that the directory 'foodwebs_mat' exists
    if not os.path.exists('foodwebs_mat'):
        os.makedirs('foodwebs_mat')

    # Save the .mat file using the full path including the 'foodwebs_mat/' directory
    sio.savemat(csv_file_path_dest, mat_data)

    print(f"MAT file saved as {csv_file_path_dest}")
