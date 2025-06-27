import pandas as pd

# Load the dataset
file_path = 'data/traffic_signal_dataset.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the dataframe
print("First 5 rows of the dataset:")
print(df.head())
print("\\n" + "="*50 + "\\n")

# Get a summary of the dataframe
print("DataFrame Info:")
df.info()
print("\\n" + "="*50 + "\\n")

# Check for missing values in each column
print("Missing values per column:")
print(df.isnull().sum()) 