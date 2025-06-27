import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1. Load and Preprocess Data
#--------------------------------
df = pd.read_csv('data/traffic_signal_dataset.csv')

# Feature Engineering
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Hour_of_Day'] = df['Timestamp'].dt.hour
df['Minute_of_Hour'] = df['Timestamp'].dt.minute
df = pd.get_dummies(df, columns=['Intersection_ID', 'Day_of_Week'], drop_first=True)

# Encode Target Variable
label_encoder = LabelEncoder()
df['Signal_Status_Encoded'] = label_encoder.fit_transform(df['Signal_Status'])

# Prepare final DataFrame
df_processed = df.drop(['Timestamp', 'Signal_Status'], axis=1)

# 2. Prepare Data for PyTorch
#--------------------------------
# Separate features (X) and target (y)
X = df_processed.drop('Signal_Status_Encoded', axis=1)
y = df_processed['Signal_Status_Encoded']

# Split data into training (70%), validation (15%), and testing (15%) sets
# First, split into training + validation (85%) and testing (15%)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

# Now, split the train_val set into training (70%) and validation (15%)
# The new test_size should be relative to the 85% block, so 0.15 / 0.85
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=(0.15/0.85), random_state=42, stratify=y_train_val)

# Scale numerical features
# Note: Boolean columns from get_dummies are already 0/1, so we focus on the others.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Convert data to PyTorch Tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
X_val_tensor = torch.FloatTensor(X_val_scaled)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_train_tensor = torch.LongTensor(y_train.values)
y_val_tensor = torch.LongTensor(y_val.values)
y_test_tensor = torch.LongTensor(y_test.values)


# 3. Define the Neural Network
#--------------------------------
class TrafficClassifier(nn.Module):
    def __init__(self, input_features):
        super(TrafficClassifier, self).__init__()
        self.layer_1 = nn.Linear(input_features, 128)
        self.layer_2 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 3) # 3 classes: Red, Green, Yellow
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.dropout(x)
        x = self.relu(self.layer_2(x))
        x = self.dropout(x)
        x = self.output_layer(x)
        return x

# 4. Train the Model
#--------------------------------
input_dim = X_train_tensor.shape[1]
model = TrafficClassifier(input_features=input_dim)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01) # Using our best learning rate

# Training loop
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass on training data
    train_outputs = model(X_train_tensor)
    train_loss = criterion(train_outputs, y_train_tensor)
    
    # Backward pass and optimization
    train_loss.backward()
    optimizer.step()
    
    # --- Validation ---
    if (epoch+1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
            _, predicted_val = torch.max(val_outputs.data, 1)
            total_val = y_val_tensor.size(0)
            correct_val = (predicted_val == y_val_tensor).sum().item()
            val_accuracy = 100 * correct_val / total_val

        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Accuracy: {val_accuracy:.2f}%')


# 5. Evaluate the Model on the Test Set
#--------------------------------
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    _, predicted = torch.max(test_outputs.data, 1)
    
    total = y_test_tensor.size(0)
    correct = (predicted == y_test_tensor).sum().item()
    
    accuracy = 100 * correct / total
    print(f'\\nFinal Accuracy on TEST data: {accuracy:.2f}%') 