import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import keras
from keras import layers

keras.utils.set_random_seed(42)

# Load original training dataset
train_data_no_aug = keras.utils.image_dataset_from_directory(
    "train",
    image_size=(150, 150),
    batch_size=32,
    label_mode="binary"
)

# Load testing dataset
test_data_no_aug = keras.utils.image_dataset_from_directory(
    "test",
    image_size=(150, 150),
    batch_size=32,
    label_mode="binary",
    shuffle=False
)

# Define augmentation layers
augmenter = keras.Sequential([
    layers.RandomRotation(40 / 360, fill_mode="nearest"),
    layers.RandomTranslation(0.2, 0.2, fill_mode="nearest"),
    layers.RandomFlip("horizontal"),
])

# Generate augmented training batches
train_data_gen = train_data_no_aug.map(
    lambda x, y: (augmenter(x, training=True), y)
)

# Build CNN model
def build_model():
    model = keras.Sequential([
        keras.Input(shape=(150, 150, 3)),
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

# Train baseline model
model = build_model()

model.fit(
    train_data_no_aug,
    epochs=3,
    verbose=2
)

loss, accuracy = model.evaluate(test_data_no_aug, verbose=0)

print(f"Test accuracy without augmentation: {accuracy:.4f}")

# Train augmented model
model_aug = build_model()

model_aug.fit(
    train_data_gen,
    epochs=3,
    verbose=2
)

loss_aug, accuracy_aug = model_aug.evaluate(
    test_data_no_aug,
    verbose=0
)

print(f"Test accuracy with augmentation: {accuracy_aug:.4f}")
