import tensorflow as tf
from tensorflow.keras.applications import VGG16

# Load VGG16 pre-trained on ImageNet
# include_top=False removes the original classification layers
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Display the model architecture
base_model.summary()
