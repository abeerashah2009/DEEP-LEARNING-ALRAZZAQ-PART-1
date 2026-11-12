import json
import keras
from keras import layers

# Load MNIST dataset
(x_train, y_train), _ = keras.datasets.mnist.load_data()

# Use 2000 images and normalize pixel values
x_train = x_train[:2000].reshape(-1, 784).astype("float32") / 255.0
y_train = y_train[:2000]

# Build a simple neural network
model = keras.Sequential([
    keras.Input(shape=(784,)),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax"),
])

# Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the model for 30 epochs
history = model.fit(
    x_train,
    y_train,
    epochs=30,
    validation_split=0.2,
    verbose=2
)

# Display available training metrics
print(history.history.keys())

# Save training history as JSON
with open("history.json", "w") as f:
    json.dump(history.history, f)

print("Training history saved successfully!")
