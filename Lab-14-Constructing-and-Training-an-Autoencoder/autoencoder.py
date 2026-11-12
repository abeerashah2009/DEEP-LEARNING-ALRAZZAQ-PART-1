import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Load the MNIST dataset
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values between 0 and 1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Flatten 28x28 images into 784-dimensional vectors
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

print("train:", x_train.shape, "test:", x_test.shape)

# Define autoencoder architecture
input_dim = x_train.shape[1]
encoding_dim = 32

input_img = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation="relu")(input_img)
decoded = Dense(input_dim, activation="sigmoid")(encoded)

autoencoder = Model(input_img, decoded)

# Display model architecture
autoencoder.summary()

# Compile the autoencoder
autoencoder.compile(
    optimizer="adam",
    loss="binary_crossentropy"
)

# Train the autoencoder
history = autoencoder.fit(
    x_train,
    x_train,
    epochs=5,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test, x_test),
    verbose=2
)

print("final loss:", history.history["loss"][-1])
print("final val_loss:", history.history["val_loss"][-1])

# Reconstruct test images
decoded_imgs = autoencoder.predict(x_test, verbose=0)

# Calculate reconstruction mean squared error
mse = float(np.mean((x_test - decoded_imgs) ** 2))
print("test reconstruction MSE:", mse)

# Compare original and reconstructed images
n = 10
plt.figure(figsize=(20, 4))

for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.savefig("reconstruction.png")
print("saved reconstruction.png")
