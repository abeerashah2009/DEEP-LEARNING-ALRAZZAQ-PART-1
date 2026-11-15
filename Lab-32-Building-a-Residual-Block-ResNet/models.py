from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    ReLU,
    MaxPooling2D,
    GlobalAveragePooling2D,
    Dense
)

from blocks import residual_block


def create_resnet(input_shape=(32, 32, 3), num_classes=10):
    """
    Create a simple ResNet model using two residual blocks.
    """

    inputs = Input(shape=input_shape)

    # Initial convolution
    x = Conv2D(
        64,
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=False
    )(inputs)

    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Two residual blocks
    x = residual_block(x, filters=64, stride=1)
    x = residual_block(x, filters=64, stride=1)

    # Classification head
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    return Model(inputs, outputs, name="Simple_ResNet")


def create_standard_cnn(input_shape=(32, 32, 3), num_classes=10):
    """
    Create a standard CNN without residual connections.
    """

    inputs = Input(shape=input_shape)

    # First convolution
    x = Conv2D(
        32,
        kernel_size=3,
        padding="same",
        use_bias=False
    )(inputs)

    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Second convolution
    x = Conv2D(
        64,
        kernel_size=3,
        padding="same",
        use_bias=False
    )(x)

    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Pooling and classification
    x = MaxPooling2D(pool_size=2)(x)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    return Model(inputs, outputs, name="Standard_CNN")
