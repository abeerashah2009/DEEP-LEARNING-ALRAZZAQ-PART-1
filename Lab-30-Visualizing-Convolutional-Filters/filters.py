import matplotlib
matplotlib.use('Agg')  # Use non-interactive mode on the server

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications import VGG16

# Load the pre-trained VGG16 model
model = VGG16(weights='imagenet', include_top=False)

# Select the first convolutional layer
layer_name = 'block1_conv1'
layer_model = model.get_layer(name=layer_name)

# Extract filters and biases
filters, biases = layer_model.get_weights()

# Display filter dimensions
print('filters.shape', filters.shape)
# Normalize filters to the range 0-1
f_min, f_max = filters.min(), filters.max()

if f_max > f_min:
    filters = (filters - f_min) / (f_max - f_min)
else:
    filters = np.zeros_like(filters)

# Select six filters for visualization
n_filters = 6
index = 1

plt.figure(figsize=(4, 8))

for i in range(n_filters):
    f = filters[:, :, :, i]

    for j in range(3):  # RGB channels
        plt.subplot(n_filters, 3, index)
        plt.imshow(f[:, :, j], cmap='gray')
        plt.axis('off')
        index += 1

# Save the visualization
plt.savefig('filters.png')
plt.close()

print('Filter visualization saved as filters.png')
