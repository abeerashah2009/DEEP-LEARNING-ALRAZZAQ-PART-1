
import numpy as np
import tensorflow as tf
import keras


def custom_mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


# Under-prediction costs 3 times more
def asymmetric_mse(y_true, y_pred):
    err = y_true - y_pred
    weight = tf.where(err > 0, 3.0, 1.0)
    return tf.reduce_mean(weight * tf.square(err))

def build_model():
    keras.utils.set_random_seed(42)

    return keras.Sequential([
        keras.Input(shape=(10,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1, activation='linear'),
    ])
model_custom = build_model()
model_custom.compile(optimizer='adam', loss=custom_mse)

model_builtin = build_model()
model_builtin.compile(optimizer='adam', loss='mse')
rng = np.random.default_rng(0)

X_train = rng.random((1000, 10)).astype('float32')
y_train = X_train.sum(axis=1, keepdims=True)

X_val = rng.random((200, 10)).astype('float32')
y_val = X_val.sum(axis=1, keepdims=True)
print('--- custom_mse ---')
model_custom.fit(
    X_train,
    y_train,
    epochs=10,
    validation_data=(X_val, y_val),
    verbose=2
)

print('--- built-in mse ---')
model_builtin.fit(
    X_train,
    y_train,
    epochs=10,
    validation_data=(X_val, y_val),
    verbose=2
)
print('Custom validation loss:  ', model_custom.evaluate(X_val, y_val, verbose=0))
print('Built-in validation loss:', model_builtin.evaluate(X_val, y_val, verbose=0))

print('--- asymmetric_mse ---')
model_asym = build_model()
model_asym.compile(optimizer='adam', loss=asymmetric_mse)

model_asym.fit(
    X_train,
    y_train,
    epochs=10,
    validation_data=(X_val, y_val),
    verbose=2
)

pred = model_asym.predict(X_val, verbose=0)

print('Asymmetric validation loss:', model_asym.evaluate(X_val, y_val, verbose=0))
print('Mean error (true - pred):', float((y_val - pred).mean()))
