# Import Matplotlib and configure it for a headless environment
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

# Set a random seed for reproducible results
tf.keras.utils.set_random_seed(42)
# Load the MNIST handwritten digit dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize images and add the channel dimension
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255.0
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255.0

# Convert labels into one-hot encoded format
train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)

# Display dataset shapes
print("Train:", train_images.shape, "Test:", test_images.shape)
# Build the Convolutional Neural Network
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Display model architecture
model.summary()
# Compile the CNN model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
# Train the CNN model
history = model.fit(
    train_images,
    train_labels,
    epochs=5,
    batch_size=64,
    validation_split=0.1,
    verbose=2
)
# Evaluate the model using test data
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'Test accuracy: {test_acc:.4f}')
print(f'Test loss: {test_loss:.4f}')

# Save the trained model
model.save('mnist_cnn.keras')
# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('CNN Training and Validation Accuracy')

plt.ylim([0, 1])
plt.legend(loc='lower right')

# Save the graph
plt.savefig('history.png')
print("Accuracy graph saved as history.png")
