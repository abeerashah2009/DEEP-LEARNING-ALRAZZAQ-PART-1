import keras
from keras.models import model_from_json

# Load model architecture from JSON
with open('model_architecture.json', 'r') as json_file:
    loaded_model = model_from_json(json_file.read())

# Restore trained weights
loaded_model.load_weights('model.weights.h5')

# Load the complete model
full_model = keras.saving.load_model('model.keras')

# Display model architecture and parameters
print("JSON + Weights Model:")
loaded_model.summary()

print("\nComplete .keras Model:")
full_model.summary()
