import pandas as pd
import os
import networkx as nx

# Function to calculate the connectance of a network
def calculate_connectance(network):
    """
    Calculate the connectance of the network (ratio of existing links to all possible links).
    """
    num_nodes = network.number_of_nodes()
    if num_nodes > 1:
        # For directed networks, the possible links are n * (n-1)
        n_possible_links = num_nodes * (num_nodes - 1)  # Directed graph possible links
        n_existing_links = network.number_of_edges()
        connectance = n_existing_links / n_possible_links
    else:
        connectance = 0  # If the network has 1 or fewer nodes, connectance is 0
    return connectance

# Load the original dataset
original_csv_path = '283_2_FoodWebDataBase_2018_12_10 (Original).csv'
original_data = pd.read_csv(original_csv_path, low_memory=False)

# Extract the unique food webs
foodwebs = original_data['foodweb.name'].unique()

# Create a directory to save the individual food web CSV files
output_dir = 'foodwebs/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Dictionary to store the number of nodes, edges, and connectance for each food web
foodweb_metrics = {}

# Iterate over each food web, filter the corresponding rows, and save them to separate CSV files
for foodweb in foodwebs:
    # Filter the data for the current food web
    foodweb_data = original_data[original_data['foodweb.name'] == foodweb]
    
    # Create a network using consumer ('con.taxonomy') and resource ('res.taxonomy') columns
    consumers = foodweb_data['con.taxonomy']
    resources = foodweb_data['res.taxonomy']
    
    # Combine consumers and resources into edges and create a NetworkX graph
    edges = list(zip(consumers, resources))
    foodweb_network = nx.DiGraph(edges)  # Directed graph (food webs are often directed)

    # Get the number of nodes and edges
    num_nodes = foodweb_network.number_of_nodes()
    num_edges = foodweb_network.number_of_edges()

    # Calculate the connectance
    connectance = calculate_connectance(foodweb_network)

    # Store the number of nodes, edges, and connectance in the dictionary
    foodweb_metrics[foodweb] = {'Nodes': num_nodes, 'Edges': num_edges, 'Connectance': connectance}

# Print the number of nodes, edges, and connectance for each food web
for foodweb, metrics in foodweb_metrics.items():
    print(f"Foodweb: {foodweb}, Nodes: {metrics['Nodes']}, Edges: {metrics['Edges']}, Connectance: {metrics['Connectance']}")

# Save the number of nodes, edges, and connectance to a CSV file
metrics_output_path = os.path.join(output_dir, 'foodweb_metrics.csv')
metrics_df = pd.DataFrame.from_dict(foodweb_metrics, orient='index').reset_index()
metrics_df.columns = ['Foodweb', 'Nodes', 'Edges', 'Connectance']
metrics_df.to_csv(metrics_output_path, index=False)
