import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import keras
from keras import layers


def build_model():
    model = keras.Sequential([
        keras.Input(shape=(64,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
