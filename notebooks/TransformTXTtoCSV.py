import re

# Input and output file paths
input_file_path = '/Users/jorge/Documents/MATLAB/data/result/Grand Caricaie  marsh dominated by Cladietum marisci, mown  Clmown1.txt'
output_file_path = '/Users/jorge/Documents/MATLAB/data/result/Grand Caricaie  marsh dominated by Cladietum marisci, mown  Clmown1.csv'

# Read the content from the input file
with open(input_file_path, 'r') as file:
    lines = file.readlines()

# Prepare the output CSV content
csv_lines = []
header_added = False

for line in lines:
    # Remove the header separators and skip them
    if re.match(r'\|={10,}\|', line):
        continue
    
    # Extract header and replace '|' with ',' only for the first header occurrence
    if not header_added and '|' in line:
        header = line.strip().replace('|', ',').strip(',')
        csv_lines.append(header)
        header_added = True
        continue
    
    # Extract data rows and replace '|' with ','
    if header_added and '|' in line:
        data_row = line.strip().replace('|', ',').strip(',')
        csv_lines.append(data_row)

# Write the CSV content to the output file
with open(output_file_path, 'w') as file:
    for csv_line in csv_lines:
        file.write(csv_line + '\n')

print(f"CSV file has been created: {output_file_path}")
