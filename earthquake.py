# earthquake.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import time

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

# Load dataset
data = pd.read_csv("database.csv")

# Select important columns
data = data[['Date', 'Time', 'Latitude', 'Longitude', 'Depth', 'Magnitude']]

# Convert Date + Time → Timestamp
timestamp = []

for d, t in zip(data['Date'], data['Time']):
    try:
        ts = datetime.datetime.strptime(d + ' ' + t, '%m/%d/%Y %H:%M:%S')
        timestamp.append(time.mktime(ts.timetuple()))
    except:
        timestamp.append(np.nan)

data['Timestamp'] = timestamp

# Remove invalid rows
data = data.dropna()

# Prepare input and output
X = data[['Timestamp', 'Latitude', 'Longitude']]
y = data[['Magnitude', 'Depth']]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build Model
model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(3,)))
model.add(Dense(16, activation='relu'))
model.add(Dense(2, activation='linear'))

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# Train Model
model.fit(X_train, y_train, epochs=10, batch_size=10)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", acc)
