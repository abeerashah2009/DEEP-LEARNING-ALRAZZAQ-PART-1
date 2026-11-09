# Lab 2: Setting Up Your Deep Learning Environment

import numpy as np
import tensorflow as tf
import keras

print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)

# Create a simple neural network
model = keras.Sequential([
    keras.Input(shape=(1,)),
    keras.layers.Dense(units=1)
])

# Configure the model
model.compile(
    optimizer='sgd',
    loss='mean_squared_error'
)

# Training data
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# Train the model
model.fit(xs, ys, epochs=500, verbose=0)

# Make a prediction
prediction = model.predict(np.array([[10.0]]), verbose=0)
value = float(prediction[0][0])

print("Prediction for 10.0:", value)
print("PASS" if 18.5 < value < 19.5 else "FAIL")
