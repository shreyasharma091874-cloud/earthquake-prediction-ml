# earthquake_lstm.py

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import datetime
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -------------------------------
# 1. LOAD DATA
# -------------------------------
data = pd.read_csv("database.csv")

data = data[['Date', 'Time', 'Latitude', 'Longitude', 'Depth', 'Magnitude']]

# -------------------------------
# 2. CONVERT TO TIMESTAMP
# -------------------------------
timestamps = []

for d, t in zip(data['Date'], data['Time']):
    try:
        ts = datetime.datetime.strptime(d + ' ' + t, '%m/%d/%Y %H:%M:%S')
        timestamps.append(time.mktime(ts.timetuple()))
    except:
        timestamps.append(np.nan)

data['Timestamp'] = timestamps
data = data.dropna()

# -------------------------------
# 3. FEATURES & LABELS
# -------------------------------
X = data[['Timestamp', 'Latitude', 'Longitude']].values
y = data[['Magnitude', 'Depth']].values

print("Original data size:", len(X))

# -------------------------------
# 4. NORMALIZATION
# -------------------------------
x_scaler = StandardScaler()
y_scaler = StandardScaler()

X = x_scaler.fit_transform(X)
y = y_scaler.fit_transform(y)

# -------------------------------
# 5. CREATE SEQUENCES (AUTO FIX)
# -------------------------------
def create_sequences(X, y, seq_length=2):

    # auto adjust if dataset small
    if len(X) <= seq_length:
        seq_length = 1

    Xs, ys = [], []

    for i in range(len(X) - seq_length):
        Xs.append(X[i:i+seq_length])
        ys.append(y[i+seq_length])

    return np.array(Xs), np.array(ys)

X, y = create_sequences(X, y, seq_length=2)

print("After sequence:", len(X))

# SAFETY CHECK
if len(X) == 0:
    print("❌ Dataset too small! Please add more data.")
    exit()

# -------------------------------
# 6. TRAIN TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Convert to tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)

# -------------------------------
# 7. LSTM MODEL
# -------------------------------
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]   # last timestep
        out = self.fc(out)
        return out

model = LSTMModel()

# -------------------------------
# 8. LOSS + OPTIMIZER
# -------------------------------
criterion = nn.HuberLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -------------------------------
# 9. TRAINING
# -------------------------------
epochs = 15

for epoch in range(epochs):
    model.train()

    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# -------------------------------
# 10. TESTING
# -------------------------------
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    test_loss = criterion(predictions, y_test)

print("Test Loss:", test_loss.item())
