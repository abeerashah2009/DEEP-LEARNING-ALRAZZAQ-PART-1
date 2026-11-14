import keras
from keras.callbacks import ModelCheckpoint

from model_def import build_model
from data import X_train, y_train

# Set random seed for reproducibility
keras.utils.set_random_seed(42)

# Build the neural network
model = build_model()

# Save the best model weights based on validation accuracy
checkpoint = ModelCheckpoint(
    "best.weights.h5",
    monitor="val_accuracy",
    save_best_only=True,
    save_weights_only=True,
    mode="max",
    verbose=1
)

# Train the model
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=10,
    callbacks=[checkpoint],
    verbose=2
)

print("Best val_accuracy:", max(history.history["val_accuracy"]))
