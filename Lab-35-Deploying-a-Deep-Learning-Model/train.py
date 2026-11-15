import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten

# Load the MNIST dataset
try:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
except Exception as e:
    sys.exit(
        f"Could not download MNIST. "
        f"Check internet access to storage.googleapis.com: {e}"
    )

# Normalize pixel values between 0 and 1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Build the neural network
model = Sequential([
    Input(shape=(28, 28)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train for exactly 2 epochs
model.fit(
    x_train,
    y_train,
    epochs=2,
    verbose=2
)

# Save the Keras model
model.save('my_model.keras')

# Export the SavedModel format
model.export('saved_model_dir')

# Display model architecture
model.summary()

# Save architecture as JSON
with open('model.json', 'w') as f:
    f.write(model.to_json())

# Save reference predictions for comparison
np.save(
    'ref_preds.npy',
    model.predict(x_test[:100], verbose=0)
)

print("\nTraining and model export completed successfully.")
