import numpy as np
import keras
from keras import layers

# Set reproducible random seed
keras.utils.set_random_seed(0)

# Build a simple neural network
model = keras.Sequential([
    keras.Input(shape=(784,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax'),
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy'
)

# Generate sample training data
x_train = np.random.random((256, 784)).astype('float32')
y_train = np.random.randint(0, 10, size=(256,))

# Train the model
model.fit(x_train, y_train, epochs=2, verbose=0)

# Save model architecture as JSON
with open('model_architecture.json', 'w') as json_file:
    json_file.write(model.to_json())

# Generate and save test data
x_test = np.random.random((5, 784)).astype('float32')
np.save('x_test.npy', x_test)

# Save original predictions
original_predictions = model.predict(x_test, verbose=0)
np.save('original_pred.npy', original_predictions)

# Save model weights in HDF5 format
model.save_weights('model.weights.h5')

# Save complete model in Keras format
model.save('model.keras')

print('Model architecture, weights, and complete model saved successfully.')
