import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = 'data/traffic_signal_dataset.csv'
df = pd.read_csv(file_path)

# --- Feature Engineering from Timestamp ---
# Convert 'Timestamp' to datetime objects
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Extract features
df['Hour_of_Day'] = df['Timestamp'].dt.hour
df['Minute_of_Hour'] = df['Timestamp'].dt.minute

# --- Encode Categorical Features ---
# One-hot encode Intersection_ID and Day_of_Week
df = pd.get_dummies(df, columns=['Intersection_ID', 'Day_of_Week'], drop_first=True)

# --- Encode the Target Variable ---
# Use LabelEncoder for 'Signal_Status'
label_encoder = LabelEncoder()
df['Signal_Status_Encoded'] = label_encoder.fit_transform(df['Signal_Status'])

# --- Final Steps ---
# Drop original columns that are no longer needed
df_processed = df.drop(['Timestamp', 'Signal_Status'], axis=1)

# Display the mapping for the encoded target variable
print("Signal Status Label-Encoding Mapping:")
for i, class_name in enumerate(label_encoder.classes_):
    print(f"{class_name} -> {i}")
print("\\n" + "="*50 + "\\n")


# Display the first few rows of the processed dataframe
print("First 5 rows of the processed dataset:")
print(df_processed.head())
print("\\n" + "="*50 + "\\n")

# Display the new dataframe info
print("Processed DataFrame Info:")
df_processed.info() 