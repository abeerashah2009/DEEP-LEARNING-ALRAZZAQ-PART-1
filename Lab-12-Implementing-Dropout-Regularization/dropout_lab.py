import numpy as np
import keras
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import layers

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Set random seed for reproducibility
keras.utils.set_random_seed(42)

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Use first 10,000 training images
x_train, y_train = x_train[:10000], y_train[:10000]

# Reshape and normalize images
x_train = x_train.reshape(
    (x_train.shape[0], 28, 28, 1)
).astype("float32") / 255

x_test = x_test.reshape(
    (x_test.shape[0], 28, 28, 1)
).astype("float32") / 255

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print("Training data:", x_train.shape, y_train.shape)
print("Testing data:", x_test.shape, y_test.shape)


# Create CNN model with optional Dropout
def create_model(dropout=True):

    model = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),

        layers.Conv2D(
            32, (3, 3), activation="relu"
        ),

        layers.MaxPooling2D((2, 2)),

        *([layers.Dropout(0.25)] if dropout else []),

        layers.Flatten(),

        layers.Dense(128, activation="relu"),

        *([layers.Dropout(0.5)] if dropout else []),

        layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# Build both models with the same initial random seed
keras.utils.set_random_seed(42)
model_without_dropout = create_model(dropout=False)

keras.utils.set_random_seed(42)
model_with_dropout = create_model(dropout=True)

print("\n=== CNN MODEL WITH DROPOUT ===")
model_with_dropout.summary()


# ==========================================
# TASK 2: TRAIN BOTH MODELS
# ==========================================

print("\n=== Training WITHOUT dropout ===")

history_without_dropout = model_without_dropout.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=128,
    validation_data=(x_test, y_test),
    verbose=2
)


print("\n=== Training WITH dropout ===")

history_with_dropout = model_with_dropout.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=128,
    validation_data=(x_test, y_test),
    verbose=2
)


# ==========================================
# TASK 3: COMPARE MODEL PERFORMANCE
# ==========================================

def plot_results(h_no, h_do, path="dropout_comparison.png"):

    plt.figure(figsize=(12, 4))

    names = [
        "Train No Dropout",
        "Test No Dropout",
        "Train Dropout",
        "Test Dropout"
    ]

    metrics = [
        ("accuracy", "Model Accuracy", "lower right"),
        ("loss", "Model Loss", "upper right")
    ]

    for i, (metric, title, location) in enumerate(
        metrics, start=1
    ):

        plt.subplot(1, 2, i)

        plt.plot(h_no.history[metric])
        plt.plot(h_no.history["val_" + metric])

        plt.plot(h_do.history[metric])
        plt.plot(h_do.history["val_" + metric])

        plt.title(title)
        plt.ylabel(metric.capitalize())
        plt.xlabel("Epoch")

        plt.legend(names, loc=location)

    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()


# Generate comparison graph
plot_results(
    history_without_dropout,
    history_with_dropout
)


# Display final training results
print("\nFinal epoch results")

print(
    f"{'model':<14}"
    f"{'train_acc':>10}"
    f"{'val_acc':>10}"
    f"{'gap':>8}"
)

gaps = {}

for name, history in [
    ("no dropout", history_without_dropout),
    ("dropout", history_with_dropout)
]:

    train_acc = history.history["accuracy"][-1]
    val_acc = history.history["val_accuracy"][-1]

    gap = train_acc - val_acc
    gaps[name] = gap

    print(
        f"{name:<14}"
        f"{train_acc:>10.4f}"
        f"{val_acc:>10.4f}"
        f"{gap:>8.4f}"
    )


smaller = (
    "dropout"
    if gaps["dropout"] < gaps["no dropout"]
    else "no dropout"
)

print("Smaller train/validation gap:", smaller)

print("\nTraining and evaluation completed.")
