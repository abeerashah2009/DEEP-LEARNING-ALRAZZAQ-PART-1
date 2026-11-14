# Lab 27 - Hyperparameter Tuning in Deep Learning

## 1. Objective

The objective of this lab is to understand hyperparameters and learn how to improve a deep learning model by testing different configurations.

We tested different learning rates and batch sizes to identify the best-performing configuration.

## 2. Tools and Technologies

- Python 3.12.3
- TensorFlow 2.17.0
- Keras
- MNIST Dataset
- Ubuntu 24.04
- Python Virtual Environment

## 3. Hyperparameters

Two hyperparameters were tested:

### Learning Rate

The learning rate controls how much the model updates its weights during training.

Values tested:
- 0.001
- 0.01
- 0.1

### Batch Size

Batch size determines how many training samples are processed before the model updates its weights.

Values tested:
- 16
- 32
- 64

A total of 9 combinations were tested.

## 4. Dataset

The MNIST dataset contains handwritten digit images from 0 to 9.

Dataset details:
- Original training images: 60,000
- Training images used: 10,000
- Testing images: 10,000
- Image dimensions: 28 x 28 pixels
- Pixel values normalized between 0 and 1

## 5. Model Architecture

The neural network contains:

1. Input layer: 28 x 28
2. Flatten layer
3. Dense layer: 128 neurons with ReLU activation
4. Output layer: 10 neurons with Softmax activation

Optimizer: Adam

Loss function: Sparse Categorical Crossentropy

Epochs: 2

## 6. Experiment Results

| Learning Rate | Batch Size | Test Loss | Test Accuracy |
|---|---:|---:|---:|
| 0.001 | 16 | 0.2271 | 92.90% |
| 0.001 | 32 | 0.2582 | 92.27% |
| 0.001 | 64 | 0.2867 | 91.65% |
| 0.01 | 16 | 0.3016 | 92.25% |
| 0.01 | 32 | 0.2667 | 91.98% |
| 0.01 | 64 | 0.2470 | 92.40% |
| 0.1 | 16 | 1.8574 | 33.23% |
| 0.1 | 32 | 1.0232 | 74.16% |
| 0.1 | 64 | 0.7387 | 77.67% |

## 7. Best Configuration

The best result was:

- Learning Rate: 0.001
- Batch Size: 16
- Test Loss: 0.2271
- Test Accuracy: 92.90%

## 8. Observations

- Learning rate 0.001 achieved the highest test accuracy.
- Learning rate 0.1 performed poorly in comparison.
- Batch size affected model performance.
- The best result was achieved with learning rate 0.001 and batch size 16.

## 9. Conclusion

In this lab, I learned how to perform hyperparameter tuning by testing different learning rates and batch sizes.

I trained and evaluated nine neural network configurations using the MNIST dataset.

The best configuration achieved 92.90% test accuracy.

This experiment demonstrated that selecting suitable hyperparameters can improve model performance.
