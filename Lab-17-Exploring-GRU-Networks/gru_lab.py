import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import keras
from keras import layers
from sklearn.preprocessing import MinMaxScaler

keras.utils.set_random_seed(42)

# Load dataset
data = pd.read_csv('data/sample_time_series.csv')
values = data[['value']].values

# Chronological split: first 80% train, last 20% test
split = int(len(values) * 0.8)
train_raw, test_raw = values[:split], values[split:]

# Fit scaler only on training data
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_raw)
test_scaled = scaler.transform(test_raw)

def create_sequences(arr, seq_length):
    X, y = [], []
    for i in range(len(arr) - seq_length):
        X.append(arr[i:i + seq_length])
        y.append(arr[i + seq_length])
    return np.array(X), np.array(y)

sequence_length = 50
X_train, y_train = create_sequences(train_scaled, sequence_length)
X_test, y_test = create_sequences(test_scaled, sequence_length)

print('Train:', X_train.shape, 'Test:', X_test.shape)

# Build a model using the selected recurrent layer
def build_model(cell):
    model = keras.Sequential([
        keras.Input(shape=(sequence_length, 1)),
        cell(50, return_sequences=True),
        cell(50),
        layers.Dense(1),
    ])
    model.compile(
        optimizer='adam',
        loss='mean_squared_error'
    )
    return model

# Build GRU model
gru_model = build_model(layers.GRU)

print('GRU parameters:', gru_model.count_params())

# Train GRU model
print("\n===== GRU MODEL TRAINING =====")

t0 = time.time()

history = gru_model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_test, y_test),
    verbose=2
)

gru_time = time.time() - t0

# Plot training and validation losses
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('GRU Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('gru_loss.png')
plt.close()

print(f"GRU training time: {gru_time:.2f} seconds")
print("GRU loss graph saved: gru_loss.png")

# Build and train LSTM model for comparison
print("\n===== LSTM MODEL TRAINING =====")

lstm_model = build_model(layers.LSTM)

print('LSTM parameters:', lstm_model.count_params())

t0 = time.time()

lstm_history = lstm_model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_test, y_test),
    verbose=2
)

lstm_time = time.time() - t0

# Evaluate both models
gru_loss = gru_model.evaluate(X_test, y_test, verbose=0)
lstm_loss = lstm_model.evaluate(X_test, y_test, verbose=0)

print("\n===== MODEL COMPARISON =====")
print(f'GRU  test loss: {gru_loss:.5f}  training time: {gru_time:.1f}s')
print(f'LSTM test loss: {lstm_loss:.5f}  training time: {lstm_time:.1f}s')
