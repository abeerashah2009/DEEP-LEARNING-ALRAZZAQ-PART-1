import os

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers


# Build the Generator
def build_generator():
    model = tf.keras.Sequential([
        layers.Input(shape=(100,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(28 * 28, activation='tanh'),
        layers.Reshape((28, 28, 1))
    ])
    return model


generator = build_generator()
generator.summary()


# Build the Discriminator
def build_discriminator():
    model = tf.keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


discriminator = build_discriminator()
discriminator.summary()


# Load and preprocess the MNIST dataset
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()

x_train = (x_train.astype('float32') - 127.5) / 127.5
x_train = np.expand_dims(x_train, axis=-1)

print("x_train:", x_train.shape)


# Compile the Discriminator
discriminator.compile(
    optimizer=tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# Build the combined GAN model
discriminator.trainable = False

gan_input = layers.Input(shape=(100,))
generated_image = generator(gan_input)
gan_output = discriminator(generated_image)

gan = tf.keras.Model(gan_input, gan_output)

gan.compile(
    optimizer=tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
    loss='binary_crossentropy'
)

print("\nCombined GAN model:")
gan.summary()


# Train the GAN using explicit GradientTape updates
def train_gan(gan, generator, discriminator, x_train,
              iterations=2000, batch_size=64):

    discriminator.trainable = True

    d_optimizer = tf.keras.optimizers.Adam(
        learning_rate=2e-4, beta_1=0.5
    )
    g_optimizer = tf.keras.optimizers.Adam(
        learning_rate=2e-4, beta_1=0.5
    )
    loss_function = tf.keras.losses.BinaryCrossentropy()

    @tf.function
    def train_discriminator(real_images, noise):
        with tf.GradientTape() as tape:
            fake_images = generator(noise, training=False)
            fake_images = tf.stop_gradient(fake_images)

            real_output = discriminator(real_images, training=True)
            fake_output = discriminator(fake_images, training=True)

            real_labels = tf.ones_like(real_output) * 0.9
            fake_labels = tf.zeros_like(fake_output)

            d_loss = (
                loss_function(real_labels, real_output)
                + loss_function(fake_labels, fake_output)
            )

        gradients = tape.gradient(
            d_loss, discriminator.trainable_variables
        )
        d_optimizer.apply_gradients(
            zip(gradients, discriminator.trainable_variables)
        )

        accuracy = tf.reduce_mean(
            tf.cast(
                tf.concat([
                    real_output >= 0.5,
                    fake_output < 0.5
                ], axis=0),
                tf.float32
            )
        )

        return d_loss, accuracy

    @tf.function
    def train_generator(noise):
        with tf.GradientTape() as tape:
            fake_images = generator(noise, training=True)
            fake_output = discriminator(fake_images, training=False)

            target_labels = tf.ones_like(fake_output)
            g_loss = loss_function(target_labels, fake_output)

        gradients = tape.gradient(
            g_loss, generator.trainable_variables
        )
        g_optimizer.apply_gradients(
            zip(gradients, generator.trainable_variables)
        )

        return g_loss

    for it in range(iterations):
        indexes = np.random.randint(
            0, x_train.shape[0], batch_size
        )
        real_images = tf.convert_to_tensor(x_train[indexes])

        noise = tf.random.normal((batch_size, 100))
        d_loss, d_acc = train_discriminator(real_images, noise)

        noise = tf.random.normal((batch_size, 100))
        g_loss = train_generator(noise)

        if it % 200 == 0:
            print(
                f"Iteration {it}: "
                f"d_loss={d_loss.numpy():.3f} "
                f"d_acc={d_acc.numpy():.2f} "
                f"g_loss={g_loss.numpy():.3f}",
                flush=True
            )


train_gan(gan, generator, discriminator, x_train)


# Generate synthetic images
noise = np.random.normal(0, 1, (10, 100))
generated_images = generator.predict(noise, verbose=0)

print("generated:", generated_images.shape)


# Visualize generated images
def plot_generated_images(generated_images):
    plt.figure(figsize=(10, 1))

    for i in range(generated_images.shape[0]):
        plt.subplot(1, 10, i + 1)
        plt.imshow(generated_images[i, :, :, 0], cmap='gray')
        plt.axis('off')

    plt.savefig('gan.png', bbox_inches='tight')
    plt.close()


plot_generated_images(generated_images)
print("saved gan.png")
