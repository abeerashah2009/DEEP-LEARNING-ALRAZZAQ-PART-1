import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
import keras
from keras import layers
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Repeatable results
np.random.seed(42)
tf.random.set_seed(42)

# Synthetic training data
X_train = np.random.rand(500, 10)
y_train = np.sin(2 * np.pi * X_train).sum(axis=1, keepdims=True)


# Build a neural network
def build(activation):
    model = keras.Sequential([
        keras.Input(shape=(10,)),
        layers.Dense(64, activation=activation),
        layers.Dense(1),
    ])

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model


# Train models using different activation functions
histories = {}

for name in ["relu", "sigmoid", "tanh"]:
    model = build(name)

    h = model.fit(
        X_train,
        y_train,
        epochs=30,
        batch_size=32,
        verbose=0
    )

    histories[name] = h.history["loss"]

    print(
        f"{name}: first loss={h.history['loss'][0]:.4f} "
        f"final loss={h.history['loss'][-1]:.4f}"
    )
# Plot training loss comparison
for name, losses in histories.items():
    plt.plot(losses, label=name)

plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Training Loss Comparison")
plt.legend()
plt.savefig("loss.png", dpi=100)
# Compare activation functions
print("\nactivation  loss@epoch5  final_loss")

for name, losses in histories.items():
    print(f"{name:10s}  {losses[4]:11.4f}  {losses[-1]:10.4f}")

best = min(histories, key=lambda n: histories[n][-1])

print("Lowest final loss:", best)
