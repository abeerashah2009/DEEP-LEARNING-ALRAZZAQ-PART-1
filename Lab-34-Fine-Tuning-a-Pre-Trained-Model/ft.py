import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

tf.random.set_seed(42)

# VGG16 convolutional base, ImageNet weights
base = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(32, 32, 3)
)

# Freeze the complete pre-trained base
base.trainable = False

# New classification head for 10 CIFAR-10 classes
model = keras.Sequential([
    base,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax'),
])

# Display VGG16 layer names and trainable status
for i, layer in enumerate(base.layers):
    print(i, layer.name, layer.trainable)

# Display complete model architecture
model.summary()

# Count trainable weight tensors
print('Trainable weight tensors (frozen base):', len(model.trainable_weights))

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Use a smaller subset for CPU training
x_train, y_train = x_train[:10000], y_train[:10000]
x_test, y_test = x_test[:2000], y_test[:2000]

# Preprocess images for VGG16
x_train = preprocess_input(x_train.astype('float32'))
x_test = preprocess_input(x_test.astype('float32'))

# Display dataset shapes
print('Train:', x_train.shape, 'Test:', x_test.shape)

# Compile model for baseline training with frozen VGG16 base
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train classification head while VGG16 base remains frozen
model.fit(
    x_train,
    y_train,
    epochs=3,
    batch_size=64,
    validation_data=(x_test, y_test),
    verbose=2
)

# Evaluate baseline model while VGG16 base is frozen
_, initial_accuracy = model.evaluate(x_test, y_test, verbose=0)

print('Baseline (frozen base) trainable tensors:', len(model.trainable_weights))
print(f'Initial test accuracy: {initial_accuracy:.4f}')

# Unfreeze block5 only, keep block1-block4 frozen
base.trainable = True
for layer in base.layers:
    layer.trainable = layer.name.startswith('block5')

# Recompile with a low learning rate for fine-tuning
model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
print(f'Fine-tune trainable tensors: {len(model.trainable_weights)}')


# Fine-tune the model with VGG16 block5 unfrozen
model.fit(
    x_train,
    y_train,
    epochs=3,
    batch_size=64,
    validation_data=(x_test, y_test),
    verbose=2
)

# Evaluate the fine-tuned model
_, final_accuracy = model.evaluate(x_test, y_test, verbose=0)

print(f'Final test accuracy: {final_accuracy:.4f}')
print(f'Change: {final_accuracy - initial_accuracy:+.4f}')
