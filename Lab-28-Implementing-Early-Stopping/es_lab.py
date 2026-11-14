import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

keras.utils.set_random_seed(42)

def make_data(n):
    X = np.random.rand(n, 20)
    noisy_sum = X[:, :5].sum(axis=1) + np.random.normal(0, 0.7, size=n)
    y = (noisy_sum > 2.5).astype('float32').reshape(-1, 1)
    return X, y

X_train, y_train = make_data(1000)
X_val, y_val = make_data(200)

model = Sequential([
    keras.Input(shape=(20,)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
    verbose=0
)

epochs_run = len(history.history['loss'])
print('Epochs run WITH early stopping:', epochs_run)

baseline = keras.models.clone_model(model)
baseline.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history_full = baseline.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=0
)

print('Epochs run WITHOUT early stopping:', len(history_full.history['loss']))

val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f'Early stopping    - Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

b_loss, b_acc = baseline.evaluate(X_val, y_val, verbose=0)
print(f'No early stopping - Validation Loss: {b_loss:.4f}, Validation Accuracy: {b_acc:.4f}')

assert epochs_run < 100
print('epochs_run < 100:', epochs_run < 100)

plt.plot(history.history['loss'], label='Training Loss (early stopping)')
plt.plot(history.history['val_loss'], label='Validation Loss (early stopping)')
plt.plot(history_full.history['val_loss'], label='Validation Loss (no early stopping)', linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('history.png')
