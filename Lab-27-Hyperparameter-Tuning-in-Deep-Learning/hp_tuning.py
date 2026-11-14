import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import keras
from keras.datasets import mnist

# Reproducible results
keras.utils.set_random_seed(42)

# Define the hyperparameter grid
learning_rates = [0.001, 0.01, 0.1]
batch_sizes = [16, 32, 64]

# Create a list of all possible combinations (9 in total)
parameter_grid = [(lr, bs) for lr in learning_rates for bs in batch_sizes]

print("Hyperparameter Grid:")
for lr, bs in parameter_grid:
    print(f"Learning Rate: {lr}, Batch Size: {bs}")

print(f"\nTotal combinations: {len(parameter_grid)}")
# Load dataset once (downloads on first use)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values
x_train, x_test = x_train / 255.0, x_test / 255.0

# Use 10,000 training samples
x_train, y_train = x_train[:10000], y_train[:10000]

print("\nDataset Information:")
print("Training images:", x_train.shape)
print("Training labels:", y_train.shape)
print("Testing images:", x_test.shape)
print("Testing labels:", y_test.shape)
def train_model(learning_rate, batch_size):
    # Model definition
    model = keras.Sequential([
        keras.Input(shape=(28, 28)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train the model
    model.fit(
        x_train,
        y_train,
        epochs=2,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=0
    )

    # Evaluate the model
    return model.evaluate(x_test, y_test, verbose=0)
# Store experiment results
results = []

# Train a model for each hyperparameter combination
for learning_rate, batch_size in parameter_grid:
    print(f"\nTraining: LR={learning_rate}, Batch Size={batch_size}")

    loss, accuracy = train_model(learning_rate, batch_size)

    results.append((learning_rate, batch_size, accuracy))

    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
# Display results sorted by accuracy
print("\nHyperparameter Comparison:")
print(f"{'LR':>8} {'Batch':>6} {'Accuracy':>9}")

for lr, bs, acc in sorted(results, key=lambda x: x[2], reverse=True):
    print(f"{lr:>8} {bs:>6} {acc:>9.4f}")

# Identify the best configuration
best_result = max(results, key=lambda x: x[2])

print(
    f"\nBest Result: Learning Rate: {best_result[0]}, "
    f"Batch Size: {best_result[1]}, "
    f"Accuracy: {best_result[2]:.4f}"
)
