import keras
from keras import layers
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input

# Dataset location
path = 'data/flower_photos'

# Dataset configuration
args = dict(
    image_size=(224, 224),
    label_mode='categorical',
    batch_size=32,
    validation_split=0.2,
    seed=123
)

# Load training and validation datasets
train_raw = keras.utils.image_dataset_from_directory(
    path,
    subset='training',
    **args
)

test_raw = keras.utils.image_dataset_from_directory(
    path,
    subset='validation',
    **args
)

# Display class information
class_names = train_raw.class_names
num_classes = len(class_names)

print(f"num_classes={num_classes}")
print("class_names:", class_names)

# Apply VGG16 preprocessing
# Use a small subset to keep CPU training short
train_data = train_raw.map(
    lambda x, y: (preprocess_input(x), y)
).take(20).cache().repeat()

test_data = test_raw.map(
    lambda x, y: (preprocess_input(x), y)
).take(10).cache()

# Load pre-trained VGG16
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze pre-trained layers
base_model.trainable = False

# Add custom classification layers
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)

output_layer = layers.Dense(
    num_classes,
    activation='softmax'
)(x)

# Create final model
model = keras.Model(
    inputs=base_model.input,
    outputs=output_layer
)

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(
    train_data,
    epochs=3,
    steps_per_epoch=20,
    validation_data=test_data
)

# Evaluate the model
loss, accuracy = model.evaluate(test_data)

print(f"Test Accuracy: {accuracy*100:.2f}%")
