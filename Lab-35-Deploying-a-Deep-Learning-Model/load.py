import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model

# Load MNIST test data
(_, _), (x_test, y_test) = mnist.load_data()

# Normalize the first 100 images
x = (x_test[:100] / 255.0).astype('float32')

# Load the Keras model
model = load_model('my_model.keras')

# Load the TensorFlow SavedModel
saved = tf.saved_model.load('saved_model_dir')

# Load reference predictions
ref = np.load('ref_preds.npy')

# Generate predictions using both formats
p_keras = model.predict(x, verbose=0)
p_saved = saved.serve(x).numpy()

# Display prediction results
print('Prediction for first test image:', np.round(p_keras[0], 3))
print('Predicted digit:', p_keras[0].argmax(), 'actual:', y_test[0])

# Compare predictions
print(
    'Keras file matches original:',
    np.allclose(p_keras, ref, atol=1e-5)
)

print(
    'SavedModel matches original:',
    np.allclose(p_saved, ref, atol=1e-5)
)
