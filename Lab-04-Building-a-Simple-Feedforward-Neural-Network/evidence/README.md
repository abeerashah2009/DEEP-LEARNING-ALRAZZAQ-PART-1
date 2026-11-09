# Lab 04 - Evidence

## Lab Title
Building a Simple Feedforward Neural Network

## Evidence Overview

This folder is used to organize evidence for the practical tasks completed in Lab 04.

### 1. Python Environment
Verify that Python, TensorFlow, and NumPy are installed correctly.

### 2. Model Architecture
The Sequential model contains three Dense layers:

- Hidden Layer 1: 8 neurons, ReLU activation
- Hidden Layer 2: 16 neurons, ReLU activation
- Output Layer: 3 neurons, Softmax activation

### 3. Model Parameters
- First Dense layer: 40 parameters
- Second Dense layer: 144 parameters
- Output Dense layer: 51 parameters
- Total: 235 parameters

### 4. Training Results
- Epochs: 50
- Batch size: 16
- Optimizer: Adam
- Loss function: Categorical Crossentropy
- Metric: Accuracy
- Final training accuracy: 99.3%

### 5. Predictions
The first three samples have these true classes:

`[0, 2, 1]`

The model predictions correctly identify all three samples.

## Screenshot Evidence

Terminal screenshots can be added here to demonstrate:

1. Python and package versions.
2. Model architecture and parameter summary.
3. Final training accuracy.
4. Predictions for the first three samples.

## Note
The reported accuracy is measured on the same synthetic dataset used for training. It is not an independent test accuracy.
