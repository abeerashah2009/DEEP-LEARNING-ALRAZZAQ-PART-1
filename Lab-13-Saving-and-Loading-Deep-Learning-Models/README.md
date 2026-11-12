
# Lab 13: Saving and Loading Deep Learning Models

## Overview

This lab demonstrates how to save and reload deep learning models using TensorFlow and Keras.

It covers saving the model architecture in JSON format, saving trained weights in HDF5 format, and saving the complete model in Keras format.

It also verifies that the reloaded models produce the same predictions as the original model.

## Objectives

- Save model architecture as a JSON file.
- Save model weights in HDF5 format.
- Save the complete model in `.keras` format.
- Load a model architecture and restore its weights.
- Load a complete saved Keras model.
- Compare predictions to verify model consistency.

## Technologies Used

- Python 3
- TensorFlow CPU 2.17.0
- Keras 3.15.1
- NumPy
- Ubuntu 24.04

## Project Files

| File | Description |
|---|---|
| `save_model.py` | Builds, trains, and saves the model. |
| `load_model.py` | Loads the JSON architecture, weights, and complete model. |
| `verify.py` | Compares predictions from the original and loaded models. |
| `model_architecture.json` | Saved model architecture. |
| `model.weights.h5` | Saved model weights. |
| `model.keras` | Complete saved Keras model. |
| `x_test.npy` | Test input data. |
| `original_pred.npy` | Original model predictions. |

## Environment Setup

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the required packages:

```bash
pip install tensorflow-cpu==2.17.0
```

## How to Run

### 1. Save the Model

```bash
python save_model.py
```

This creates the JSON architecture, HDF5 weights, complete Keras model, test input, and original predictions.

### 2. Load the Model

```bash
python load_model.py
```

This loads the model architecture and weights separately and also loads the complete `.keras` model.

### 3. Verify Model Predictions

```bash
python verify.py
```

The script compares predictions from the original model with predictions from both reloaded models.

## Verification Results

The prediction comparison was successful:

```text
JSON+weights matches: True
.keras matches: True
```

Both loaded models produced predictions matching the original model using NumPy `allclose()`.

## Learning Outcome

After completing this lab, I learned how to:

- Save and restore deep learning model architecture.
- Save and reload trained model weights.
- Save and load a complete Keras model.
- Verify model consistency by comparing predictions.

## Conclusion

Lab 13 successfully demonstrates model persistence using JSON, HDF5, and Keras formats. Both model loading methods were verified successfully against the original model predictions.
