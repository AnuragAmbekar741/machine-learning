# ========== IMPORTS ==========
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
import os
import random

# Set random seeds for reproducibility
os.environ['PYTHONHASHSEED'] = '0'
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# ========== STEP 1: Load Dataset ==========
df = pd.read_csv('tesla_stock_data.csv')
print("Dataset loaded. Shape:", df.shape)

# ========== STEP 2: Normalize Features ==========
features_to_normalize = ['Open', 'High', 'Low', 'Volume', 'Close']
scaler = MinMaxScaler()
df_normalized = df.copy()
df_normalized[features_to_normalize] = scaler.fit_transform(df[features_to_normalize])
print("Features normalized.")

# ========== STEP 3: Create Time Series Sequences ==========
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, :])
        y.append(data[i, 4])  # Index 4 is 'Close'
    return np.array(X), np.array(y)

feature_columns = ['Open', 'High', 'Low', 'Volume', 'Close']
data = df_normalized[feature_columns].values
sequence_length = 30
X, y = create_sequences(data, sequence_length)
print(f"Sequences created. X shape: {X.shape}, y shape: {y.shape}")

# ========== STEP 4: Split Dataset ==========
split_idx = int(len(X) * 0.8)
X_train = X[:split_idx]
X_test = X[split_idx:]
y_train = y[:split_idx]
y_test = y[split_idx:]
print(f"Train/Test split: Train={len(X_train)}, Test={len(X_test)}")

# ========== STEP 5 & 6: Build and Compile Model ==========
tf.random.set_seed(42)
model = Sequential([
    GRU(80, input_shape=(sequence_length, len(feature_columns))),
    Dense(1)
])
model.compile(optimizer=Adam(), loss='mean_squared_error')
print("Model built and compiled.")

# ========== STEP 7: Train Model ==========
history = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# ========== STEP 8: Make Predictions ==========
y_pred = model.predict(X_test, verbose=1)

# ========== STEP 9: Evaluate Model ==========
rmse = math.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")