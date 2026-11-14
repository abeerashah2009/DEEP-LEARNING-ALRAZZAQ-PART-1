import keras

from model_def import build_model
from data import X_test, y_test

# Build a model with random weights
keras.utils.set_random_seed(7)
model = build_model()

# Evaluate before loading saved weights
before = model.evaluate(X_test, y_test, verbose=0)
print(f"Before loading weights: {before[1] * 100:.2f}%")

# Load the best saved weights
model.load_weights("best.weights.h5")

# Evaluate after loading saved weights
after = model.evaluate(X_test, y_test, verbose=0)
print(f"After loading weights:  {after[1] * 100:.2f}%")
