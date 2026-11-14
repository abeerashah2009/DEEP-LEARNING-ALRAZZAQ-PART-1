import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import keras
from keras import layers

img = keras.utils.load_img("train/a/img_000.png")

img_array = keras.utils.img_to_array(img)

batch = np.expand_dims(img_array, 0)

print("Image shape:", img_array.shape)

rotate = layers.RandomRotation(40 / 360, fill_mode="nearest")

flip = layers.RandomFlip("horizontal_and_vertical")

shift = layers.RandomTranslation(0.2, 0.2, fill_mode="nearest")

augmenter = keras.Sequential([
    rotate,
    flip,
    shift
])

out = augmenter(batch, training=True)

print("Augmented shape:", out.shape)

augmented_images = [
    augmenter(batch, training=True)[0].numpy().astype(np.uint8)
    for _ in range(5)
]

fig, axes = plt.subplots(1, 6, figsize=(20, 4))

axes[0].imshow(img_array.astype(np.uint8))
axes[0].set_title("Original Image")

for i, aug_img in enumerate(augmented_images):
    axes[i + 1].imshow(aug_img)
    axes[i + 1].set_title(f"Augmented {i+1}")

for ax in axes:
    ax.axis("off")

plt.savefig("aug.png", bbox_inches="tight")

print("saved aug.png")
