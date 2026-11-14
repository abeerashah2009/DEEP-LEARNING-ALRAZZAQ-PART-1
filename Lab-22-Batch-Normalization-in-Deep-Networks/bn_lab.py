import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

tf.keras.utils.set_random_seed(42)

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

print("Train:", x_train.shape, "Test:", x_test.shape)

# Build a neural network with Batch Normalization
model = tf.keras.Sequential([
    tf.keras.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Display the model architecture
model.summary()

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model with Batch Normalization
batch_norm_history = model.fit(
    x_train,
    y_train,
    epochs=3,
    batch_size=128,
    validation_data=(x_test, y_test),
    verbose=2
)

# Evaluate the model on the test dataset
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

print(f'\nTest accuracy with Batch Normalization: {test_acc:.4f}')

# Reset the random seed for a fair comparison
tf.keras.utils.set_random_seed(42)

# Build the model without Batch Normalization
model_without_bn = tf.keras.Sequential([
    tf.keras.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model_without_bn.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model without Batch Normalization
no_batch_norm_history = model_without_bn.fit(
    x_train,
    y_train,
    epochs=3,
    batch_size=128,
    validation_data=(x_test, y_test),
    verbose=2
)

# Evaluate the model without Batch Normalization
test_loss_no_bn, test_acc_no_bn = model_without_bn.evaluate(
    x_test, y_test, verbose=0
)

print(f'\nTest accuracy without Batch Normalization: {test_acc_no_bn:.4f}')

# Compare validation accuracy
print("\nepoch  val_acc_with_BN  val_acc_without_BN")

for i, (a, b) in enumerate(zip(
    batch_norm_history.history['val_accuracy'],
    no_batch_norm_history.history['val_accuracy']
), 1):
    print(f"{i:>5}  {a:>15.4f}  {b:>18.4f}")
