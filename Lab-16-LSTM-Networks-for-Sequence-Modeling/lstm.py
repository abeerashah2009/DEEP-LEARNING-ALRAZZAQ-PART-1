import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
import keras

from keras.models import Sequential
from keras.layers import Dense, LSTM

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)
# LSTM Model Architecture

n_steps = 3
n_features = 1

model = Sequential([
    keras.Input(shape=(n_steps, n_features)),
    LSTM(units=50),
    Dense(1),
])

model.compile(
    optimizer="adam",
    loss="mse"
)

print("LSTM model created successfully!")
model.summary()
# Dataset Preparation

data = np.sin(np.linspace(0, 50, 500))

X = np.array([
    data[i:i + n_steps]
    for i in range(len(data) - n_steps)
])

y = data[n_steps:]

X = X.reshape((X.shape[0], n_steps, n_features))

# Split dataset into training and testing sets

split = int(len(X) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print("X_train:", X_train.shape, "X_test:", X_test.shape)
print("y_train:", y_train.shape, "y_test:", y_test.shape)
# Model Training

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=16,
    verbose=0
)

print("Final training loss:", history.history["loss"][-1])
# Model Evaluation

test_mse = model.evaluate(X_test, y_test, verbose=0)

print("Test MSE:", test_mse)

predictions = model.predict(X_test, verbose=0).flatten()

print("Predictions generated:", len(predictions))
print("First 5 predictions:", predictions[:5])
# Visualization of Actual vs Predicted Data

plt.plot(range(len(data)), data, label="Actual Data")

plt.plot(
    range(n_steps + split, len(data)),
    predictions,
    label="Predicted Data (test)"
)

plt.legend()
plt.savefig("lstm.png")

print("Graph saved successfully as lstm.png")
