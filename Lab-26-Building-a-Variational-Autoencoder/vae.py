import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, Dense, Layer
from tensorflow.keras.models import Model


# Reproducibility
tf.random.set_seed(42)
np.random.seed(42)


# Model settings
input_shape = (784,)
latent_dim = 2
batch_size = 128
epochs = 50


# Sampling layer with KL-divergence loss
class Sampling(Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs

        epsilon = tf.random.normal(shape=tf.shape(z_mean))

        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(
                1 + z_log_var
                - tf.square(z_mean)
                - tf.exp(z_log_var),
                axis=1
            )
        )

        self.add_loss(kl_loss)

        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# Encoder
encoder_inputs = Input(shape=input_shape, name="encoder_input")

h = Dense(512, activation="relu")(encoder_inputs)
h = Dense(256, activation="relu")(h)

z_mean = Dense(latent_dim, name="z_mean")(h)
z_log_var = Dense(latent_dim, name="z_log_var")(h)

z = Sampling(name="sampling")([z_mean, z_log_var])

encoder = Model(
    encoder_inputs,
    [z_mean, z_log_var, z],
    name="encoder"
)


# Decoder
latent_inputs = Input(shape=(latent_dim,), name="latent_input")

decoder_h = Dense(256, activation="relu")(latent_inputs)
decoder_h = Dense(512, activation="relu")(decoder_h)

decoder_outputs = Dense(
    784,
    activation="sigmoid",
    name="decoder_output"
)(decoder_h)

decoder = Model(
    latent_inputs,
    decoder_outputs,
    name="decoder"
)


# Complete VAE
vae_outputs = decoder(encoder(encoder_inputs)[2])

vae = Model(
    encoder_inputs,
    vae_outputs,
    name="vae"
)


# Reconstruction loss
def reconstruction_loss(y_true, y_pred):
    return tf.reduce_sum(
        tf.square(y_true - y_pred),
        axis=1
    )


vae.compile(
    optimizer="adam",
    loss=reconstruction_loss
)


# Load and preprocess MNIST
print("Loading MNIST dataset...")

(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

x_train = np.reshape(x_train, [-1, 784]).astype("float32") / 255.0
x_test = np.reshape(x_test, [-1, 784]).astype("float32") / 255.0

print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)


# Train VAE
print("\nStarting VAE training...")

history = vae.fit(
    x_train,
    x_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(x_test, x_test),
    verbose=2
)


# Generate samples from the latent space
print("\nGenerating latent space visualization...")

n = 15
digit_size = 28

grid_x = np.linspace(-3, 3, n)
grid_y = np.linspace(-3, 3, n)

figure = np.zeros((digit_size * n, digit_size * n))

for i, yi in enumerate(grid_y):
    for j, xi in enumerate(grid_x):
        z_sample = np.array([[xi, yi]], dtype="float32")

        generated = decoder.predict(z_sample, verbose=0)

        digit = generated[0].reshape(digit_size, digit_size)

        figure[
            i * digit_size:(i + 1) * digit_size,
            j * digit_size:(j + 1) * digit_size
        ] = digit


plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap="gray")
plt.axis("off")
plt.title("VAE Latent Space - MNIST")
plt.tight_layout()
plt.savefig("vae_latent_grid.png", dpi=150)
plt.close()

print("\nLatent space image saved: vae_latent_grid.png")
print("VAE training and visualization completed.")
