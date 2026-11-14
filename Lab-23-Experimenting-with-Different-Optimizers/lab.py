import os, sys, json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.utils import to_categorical

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Use 10,000 training samples and normalize pixels
x_train, y_train = x_train[:10000] / 255.0, y_train[:10000]
x_test = x_test / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
# Define the neural network model
def create_model():
    return keras.Sequential([
        keras.Input(shape=(28, 28)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax'),
    ])
# Train and evaluate the model using the selected optimizer
def run(name):
    keras.utils.set_random_seed(42)

    model = create_model()

    model.compile(
        optimizer=name,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        x_train,
        y_train,
        epochs=3,
        batch_size=32,
        validation_data=(x_test, y_test),
        verbose=2
    )

    loss, acc = model.evaluate(x_test, y_test, verbose=0)

    os.makedirs('results', exist_ok=True)

    with open(f'results/{name}.json', 'w') as f:
        json.dump({
            'loss': loss,
            'accuracy': acc,
            'history': history.history
        }, f)

    print(f'{name}: loss={loss:.4f} accuracy={acc:.4f}')


# Run each optimizer passed from the command line
for opt in sys.argv[1:]:
    run(opt)
