import pandas as pd

# Load the dataset (already done, but repeated for context)
file_path = '/mnt/data/283_2_FoodWebDataBase_2018_12_10.csv'
data = pd.read_csv(file_path)

# Step 1: Handling Missing Values
# Summary of missing values
print("Missing values summary:")
print(data.isnull().sum())

# Impute numerical columns with the median
numerical_cols = data.select_dtypes(include=['number']).columns
data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].median())

# Impute categorical columns with "NA"
categorical_cols = data.select_dtypes(include=['object']).columns
data[categorical_cols] = data[categorical_cols].fillna("NA")

# Step 2: Removing Duplicate Entries
print("\nNumber of duplicate rows before removal:", data.duplicated().sum())
data = data.drop_duplicates()
print("Number of duplicate rows after removal:", data.duplicated().sum())

# Step 3: Standardizing Data Formats
# Ensure columns like 'sampling.start.year' and 'sampling.end.year' are integers
if 'sampling.start.year' in data.columns:
    data['sampling.start.year'] = pd.to_numeric(data['sampling.start.year'], errors='coerce').fillna(0).astype(int)
if 'sampling.end.year' in data.columns:
    data['sampling.end.year'] = pd.to_numeric(data['sampling.end.year'], errors='coerce').fillna(0).astype(int)

# Step 4: Outlier Detection (example with 'latitude' and 'altitude')
import numpy as np

# Replace outliers with NaN for 'latitude' (3 standard deviations from the mean)
latitude_mean = data['latitude'].mean()
latitude_std = data['latitude'].std()
data['latitude'] = data['latitude'].where(
    (np.abs(data['latitude'] - latitude_mean) <= 3 * latitude_std), np.nan
)

# Step 5: Feature Preparation
# Encoding categorical columns using label encoding as an example
from sklearn.preprocessing import LabelEncoder

# Encode columns 'interaction.classification' and 'con.taxonomy'
label_encoder = LabelEncoder()
data['interaction.classification'] = label_encoder.fit_transform(data['interaction.classification'])
data['con.taxonomy'] = label_encoder.fit_transform(data['con.taxonomy'])

# Creating a derived feature: duration between 'sampling.start.year' and 'sampling.end.year'
data['sampling_duration'] = data['sampling.end.year'] - data['sampling.start.year']

# Step 6: Final Verification
# Display the cleaned dataset summary
print("\nCleaned dataset summary:")
print(data.info())
