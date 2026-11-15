import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model

# Generate the input image
x = np.tile(np.linspace(0, 255, 224), (224, 1))
rgb = np.stack([x, x.T, 255 - x], axis=-1).astype("uint8")

img = Image.fromarray(rgb)
d = ImageDraw.Draw(img)

d.ellipse((50, 50, 150, 150), fill=(255, 255, 0))
d.rectangle((120, 130, 200, 200), outline=(0, 0, 0), width=6)

img.save("input.png")

# Load model and preprocess the image
model = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

img_array = np.expand_dims(
    np.array(img, dtype="float32"),
    axis=0
)

img_array = preprocess_input(img_array)

# Feed image through the network
feats = model.predict(img_array, verbose=0)

print("final feature map:", feats.shape)
# Task 2: Extract intermediate activations

layers = model.layers[:8]

layer_names = [layer.name for layer in layers]

activation_model = Model(
    inputs=model.input,
    outputs=[layer.output for layer in layers]
)

activations = activation_model.predict(img_array, verbose=0)

# Display layer names and activation shapes
for name, activation in zip(layer_names, activations):
    print(name, activation.shape)
# Task 3: Visualize intermediate activations

# Save the first convolutional layer activation
plt.matshow(activations[1][0, :, :, 0], cmap="viridis")
plt.title("First Convolutional Layer - Channel 0")
plt.savefig("first_layer.png")
plt.close()

# Plot channel 0 from activation indices 1 to 7
fig, axes = plt.subplots(1, 7, figsize=(20, 4))

for i, ax in enumerate(axes, start=1):
    ax.matshow(activations[i][0, :, :, 0], cmap="viridis")
    ax.set_title(layer_names[i], fontsize=8)
    ax.axis("off")

fig.savefig("activations.png")
plt.close(fig)

# Print activation statistics
print("\nActivation Statistics:")

for name, act in zip(layer_names[1:], activations[1:]):
    channel = act[0, :, :, 0]

    print(
        f"{name}: "
        f"min={channel.min():.1f} "
        f"max={channel.max():.1f} "
        f"mean={channel.mean():.1f}"
    )
