# Lab 24: Custom Loss Functions in Keras

## Objective

Understand how to create custom loss functions in Keras, integrate them into model compilation, and evaluate their impact on model performance.

## Tools and Technologies

- Python 3
- TensorFlow CPU 2.17.0
- Keras 3.15.1
- NumPy 1.26.4
- Ubuntu Linux

## Tasks Completed

### 1. Custom Mean Squared Error (MSE)

Created a custom MSE loss function using TensorFlow operations:

- Calculates the difference between actual and predicted values.
- Squares the errors.
- Calculates the mean squared error.

### 2. Model Compilation and Comparison

Compiled two neural network models:

- Model 1: Custom MSE loss function.
- Model 2: Built-in Keras MSE loss function.

Both models were trained for 10 epochs using the same training and validation datasets.

### 3. Asymmetric Mean Squared Error

Created an asymmetric loss function that applies:

- Weight 3.0 when the model under-predicts.
- Weight 1.0 when the model over-predicts or predicts correctly.

This experiment demonstrates how different error penalties can influence model predictions.

## Dataset

Generated a synthetic dataset using NumPy:

- Training samples: 1,000
- Validation samples: 200
- Input features: 10
- Target: Sum of the 10 input features

## Experimental Results

| Experiment | Validation Loss |
|---|---:|
| Custom MSE | 0.04109 |
| Built-in MSE | 0.04109 |
| Asymmetric MSE | 0.08940 |

Mean error (actual - predicted) for the asymmetric model: **-0.13246**

## Observations

1. Custom MSE produced the same validation loss as built-in MSE.
2. Both MSE models showed decreasing training and validation losses.
3. Asymmetric MSE applied a higher penalty to under-prediction.
4. The negative mean error indicates that the asymmetric model predicted slightly higher values on average.

## How to Run

Activate the virtual environment and install the required packages:

```bash
python3 -m venv venv
source venv/bin/activate
pip install tensorflow-cpu==2.17.0 "numpy<2.0"
```

Run the experiment:

```bash
python3 loss.py
```

## Main Learning Outcomes

- Creating custom loss functions in Keras.
- Using TensorFlow operations inside loss functions.
- Compiling models with custom and built-in losses.
- Comparing model performance.
- Understanding asymmetric error penalties.

## Project File

- `loss.py` — Custom loss functions, model creation, training, and evaluation.
