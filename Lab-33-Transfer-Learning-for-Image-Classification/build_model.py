import keras
from keras import layers
from keras.applications import VGG16

# Number of flower categories
num_classes = 5

# Load pre-trained VGG16 without its original classifier
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze the pre-trained layers
base_model.trainable = False

# Add our custom classification layers
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)

# Output layer for five flower categories
output_layer = layers.Dense(
    num_classes,
    activation='softmax'
)(x)

# Create the final model
model = keras.Model(
    inputs=base_model.input,
    outputs=output_layer
)

# Display model architecture
model.summary()
