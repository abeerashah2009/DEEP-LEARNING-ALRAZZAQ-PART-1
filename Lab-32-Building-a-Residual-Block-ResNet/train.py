import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from models import create_resnet, create_standard_cnn


# -----------------------------------
# 1. Load CIFAR-10 dataset
# -----------------------------------

print("Loading CIFAR-10 dataset...")

(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.cifar10.load_data()
)

# Use the required number of images
x_train = x_train[:10000]
y_train = y_train[:10000]

x_test = x_test[:2000]
y_test = y_test[:2000]

# Normalize image values between 0 and 1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print("Training images:", x_train.shape)
print("Testing images:", x_test.shape)


# -----------------------------------
# 2. Callback to measure epoch time
# -----------------------------------

class EpochTimeCallback(tf.keras.callbacks.Callback):

    def __init__(self):
        super().__init__()
        self.epoch_times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.perf_counter() - self.start_time
        self.epoch_times.append(elapsed)

        print(
            f"Epoch {epoch + 1} training time: "
            f"{elapsed:.2f} seconds"
        )


# -----------------------------------
# 3. Compile and train a model
# -----------------------------------

def train_model(model, model_name, save_path):

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    timer = EpochTimeCallback()

    print(f"\nTraining {model_name}...")

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=3,
        batch_size=64,
        callbacks=[timer]
    )

    # Save trained model
    model.save(save_path)

    print(f"{model_name} saved as {save_path}")

    return history, timer.epoch_times


# -----------------------------------
# 4. Train ResNet
# -----------------------------------

resnet = create_resnet()

resnet_history, resnet_times = train_model(
    resnet,
    "ResNet",
    "resnet.keras"
)


# -----------------------------------
# 5. Train Standard CNN
# -----------------------------------

plain_cnn = create_standard_cnn()

cnn_history, cnn_times = train_model(
    plain_cnn,
    "Standard CNN",
    "plain_cnn.keras"
)


# -----------------------------------
# 6. Compare validation accuracy
# -----------------------------------

resnet_accuracy = resnet_history.history["val_accuracy"]
cnn_accuracy = cnn_history.history["val_accuracy"]

print("\n========== FINAL COMPARISON ==========")

print(f"ResNet validation accuracy: {resnet_accuracy[-1]:.4f}")
print(f"Standard CNN validation accuracy: {cnn_accuracy[-1]:.4f}")


# -----------------------------------
# 7. Create comparison graph
# -----------------------------------

epochs = range(1, 4)

plt.figure(figsize=(12, 5))

# Validation accuracy
plt.subplot(1, 2, 1)

plt.plot(
    epochs,
    resnet_accuracy,
    marker="o",
    label="ResNet"
)

plt.plot(
    epochs,
    cnn_accuracy,
    marker="o",
    label="Standard CNN"
)

plt.title("Validation Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)


# Epoch training time
plt.subplot(1, 2, 2)

plt.plot(
    epochs,
    resnet_times,
    marker="o",
    label="ResNet"
)

plt.plot(
    epochs,
    cnn_times,
    marker="o",
    label="Standard CNN"
)

plt.title("Training Time Comparison")
plt.xlabel("Epoch")
plt.ylabel("Time (seconds)")
plt.legend()
plt.grid(True)


plt.tight_layout()
plt.savefig("comparison.png", dpi=150)

print("\nComparison graph saved as comparison.png")
print("\nSUCCESS: Both models have been trained and compared!")
