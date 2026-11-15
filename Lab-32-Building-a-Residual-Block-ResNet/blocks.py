import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    ReLU,
    Add
)


def residual_block(inputs, filters, stride=1):
    """
    Build a basic residual block.

    Args:
        inputs: Input feature map.
        filters: Number of output filters.
        stride: Convolution stride.

    Returns:
        Output feature map after residual addition and ReLU.
    """

    # Main path: first convolution
    x = Conv2D(
        filters,
        kernel_size=3,
        strides=stride,
        padding="same",
        use_bias=False
    )(inputs)

    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Main path: second convolution
    x = Conv2D(
        filters,
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=False
    )(x)

    x = BatchNormalization()(x)

    # Shortcut path
    shortcut = inputs

    input_channels = inputs.shape[-1]

    if stride != 1 or input_channels != filters:
        shortcut = Conv2D(
            filters,
            kernel_size=1,
            strides=stride,
            padding="same",
            use_bias=False
        )(shortcut)

        shortcut = BatchNormalization()(shortcut)

    # Add main path and shortcut path
    x = Add()([x, shortcut])

    # Final activation
    x = ReLU()(x)

    return x
