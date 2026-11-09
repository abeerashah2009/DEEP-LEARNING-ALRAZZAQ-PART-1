import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense

# Fix random seeds for repeatable results
np.random.seed(42)
tf.random.set_seed(42)

# Generate synthetic data
rng = np.random.default_rng(42)

centers = np.array([
    [0, 0, 0, 0],
    [3, 3, 0, 0],
    [0, 3, 3, 3]
])

y = rng.integers(0, 3, 300)
X = centers[y] + rng.normal(0, 0.7, (300, 4))
Y = keras.utils.to_categorical(y, 3)

# Task 1: Initialize a Sequential model
model = keras.Sequential([
    keras.Input(shape=(4,))
])
# Task 2: Add Dense layers

# First hidden layer: 8 neurons with ReLU activation
model.add(Dense(units=8, activation='relu'))

# Second hidden layer: 16 neurons with ReLU activation
model.add(Dense(units=16, activation='relu'))

# Output layer: 3 neurons with Softmax activation
model.add(Dense(units=3, activation='softmax'))
# Task 3: Compile and summarize the model

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Display the model architecture
model.summary()

# Train the model
model.fit(X, Y, epochs=50, batch_size=16, verbose=0)

# Evaluate model performance
loss, acc = model.evaluate(X, Y, verbose=0)

print(f"Final accuracy: {acc:.3f}")
print("True classes of first 3 samples:", y[:3])

# Display predictions for the first 3 samples
print(np.round(model.predict(X[:3], verbose=0), 3))
