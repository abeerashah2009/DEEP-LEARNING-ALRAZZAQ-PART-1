import numpy as np
import keras
from keras.models import model_from_json

# Load test input and original predictions
x_test = np.load('x_test.npy')
original_prediction = np.load('original_pred.npy')

# Reconstruct model from JSON and weights
with open('model_architecture.json') as f:
    json_model = model_from_json(f.read())

json_model.load_weights('model.weights.h5')

# Load complete Keras model
full_model = keras.saving.load_model('model.keras')

# Generate predictions
json_prediction = json_model.predict(x_test, verbose=0)
keras_prediction = full_model.predict(x_test, verbose=0)

# Display prediction samples
print('Original Prediction:', original_prediction[0][:3])
print('JSON+weights Prediction:', json_prediction[0][:3])
print('.keras Prediction:', keras_prediction[0][:3])

# Compare predictions
print(
    'JSON+weights matches:',
    np.allclose(original_prediction, json_prediction)
)

print(
    '.keras matches:',
    np.allclose(original_prediction, keras_prediction)
)
