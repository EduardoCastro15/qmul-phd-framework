import os
import pandas as pd

# Path to the CSV file that contains the list of foodweb file names
file_list_path = '/Users/jorge/Documents/MATLAB/data/foodwebs_mat/foodweb_metrics.csv'

# Load the list of foodweb file names
file_list_df = pd.read_csv(file_list_path)
foodweb_files = file_list_df['Foodweb'].tolist()

# Define a dictionary with old column names as keys and new column names as values
new_column_names = {
    'interaction.type': 'interactionType',
    'con.taxonomy': 'conTaxonomy',
    'con.mass.mean.g.': 'conMassMean',
    'res.taxonomy': 'resTaxonomy',
    'res.mass.mean.g.': 'resMassMean',
    'foodweb.name': 'foodwebName'
}

# Path to the folder where the foodweb CSV files are located
foodweb_folder_path = '/Users/jorge/Desktop/PhD/SHENIKA REDDY/foodwebs_csv'
new_foodweb_folder_path = '/Users/jorge/Desktop/PhD/SHENIKA REDDY/foodwebs_csv/no_space'

# Process each foodweb file
for foodweb_file in foodweb_files:
    # Construct the full file path
    foodweb_path = f"{foodweb_folder_path}/{foodweb_file}.csv"
    
    # Load the CSV file
    df = pd.read_csv(foodweb_path)
    
    # # Rename the columns
    # df.rename(columns=new_column_names, inplace=True)
    
    # # Add the new column 'bodyMassRatio' as the ratio of 'conMassMean' to 'resMassMean'
    # df['bodyMassRatio'] = df['conMassMean'] / df['resMassMean']

    # Modify the file name by replacing spaces with underscores
    modified_file_name = foodweb_file.replace(" ", "_") + ".csv"
    modified_foodweb_path = os.path.join(new_foodweb_folder_path, modified_file_name)

    # Save the updated DataFrame back to a CSV file (overwriting the original file)
    df.to_csv(modified_foodweb_path, index=False)
    
    print(f"File name has been updated and saved for {foodweb_file}.")
