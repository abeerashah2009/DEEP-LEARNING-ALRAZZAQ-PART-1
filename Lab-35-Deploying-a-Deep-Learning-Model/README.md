# Lab 35: Deploying a Deep Learning Model

## 1. Introduction

In this lab, I learned how to train, save, export, load, and test a Deep Learning Model using TensorFlow and Keras.

I used the MNIST dataset, which contains handwritten digits from 0 to 9.

**Main Goal:** Train a neural network and verify that the saved models produce the same predictions as the original model.

## 2. Learning Objectives

- Understand model deployment.
- Load the MNIST dataset.
- Prepare and normalize image data.
- Build a neural network using Keras.
- Train a model for 2 epochs.
- Save a model in `.keras` format.
- Export a TensorFlow SavedModel.
- Save model architecture in JSON format.
- Load saved models separately.
- Compare model predictions using NumPy.

## 3. Environment Setup

**Operating System:** Ubuntu 24.04.3 LTS

**Programming Language:** Python 3.12.3

**Libraries:**
- TensorFlow CPU: 2.17.0
- Keras: 3.15.1
- NumPy: 1.26.4

### Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Upgrade pip

```bash
pip install --upgrade pip
```

### Install Required Libraries

```bash
pip install tensorflow-cpu==2.17.0 'numpy<2'
```

## 4. Understanding the MNIST Dataset

MNIST is a dataset of handwritten digit images.

Important points:

- It contains 60,000 training images.
- It contains 10,000 testing images.
- Each image has a size of 28 × 28 pixels.
- It contains digits from 0 to 9.
- Each image has a corresponding digit label.

### What is Normalization?

Normalization means converting pixel values into a smaller range.

Original pixel values range from 0 to 255.

We divide each pixel value by 255 to convert it into a range between 0 and 1.

**Example:**

255 / 255 = 1.0

128 / 255 = approximately 0.502

This helps the neural network process image data more effectively.

## 5. Building the Neural Network

I created a Sequential Neural Network using Keras.

### Model Architecture

**Layer 1: Input Layer**
- Input image size: 28 × 28.

**Layer 2: Flatten**
- Converts the 2D image into a 1D array.
- 28 × 28 = 784 values.

**Layer 3: Dense Layer**
- 128 neurons.
- ReLU activation function.
- Learns patterns from the images.

**Layer 4: Output Dense Layer**
- 10 neurons.
- Softmax activation function.
- Predicts one of the digits from 0 to 9.

### Model Parameters

| Layer | Parameters |
|---|---:|
| Flatten | 0 |
| Dense | 100,480 |
| Dense Output | 1,290 |
| Total Trainable Parameters | 101,770 |

## 6. Model Compilation

I compiled the model using three important settings.

**Optimizer: Adam**
- Updates model weights during training.

**Loss Function: Sparse Categorical Crossentropy**
- Measures the difference between predicted and actual digit labels.

**Metrics: Accuracy**
- Measures how many predictions are correct.

## 7. Model Training

I trained the neural network for exactly 2 epochs.

### Command

```bash
python train.py
```

### Actual Training Results

| Epoch | Accuracy | Loss |
|---|---:|---:|
| 1 | 92.68% | 0.2578 |
| 2 | 96.57% | 0.1152 |

### What is an Epoch?

An epoch means one complete pass through the training dataset.

In this lab, the model completed 2 full training epochs.

**Result:** Training completed successfully.

## 8. Saving the Model

I saved the trained model in two different formats.

### A. Keras Model

File name:

`my_model.keras`

This is the native Keras model format.

Command:

```python
model.save('my_model.keras')
```

### B. TensorFlow SavedModel

Directory name:

`saved_model_dir/`

This format exports the model for TensorFlow-based use and serving.

Command:

```python
model.export('saved_model_dir')
```

### C. Model Architecture JSON

File name:

`model.json`

It stores the model architecture configuration.

Command:

```python
with open('model.json', 'w') as f:
    f.write(model.to_json())
```

### D. Reference Predictions

File name:

`ref_preds.npy`

It stores predictions for the first 100 test images.

Command:

```python
np.save('ref_preds.npy', model.predict(x_test[:100], verbose=0))
```

## 9. Loading the Saved Models

I created a separate Python script named `load.py`.

The script loads the Keras model:

```python
model = load_model('my_model.keras')
```

It also loads the TensorFlow SavedModel:

```python
saved = tf.saved_model.load('saved_model_dir')
```

This allows both saved formats to be tested separately.

**Important:** Keras 3 uses `tf.saved_model.load()` to load an exported SavedModel for inference. The regular Keras `load_model()` function does not directly load a SavedModel directory.

## 10. Model Inference and Prediction

Inference means using a trained model to make predictions on new data.

I used the first 100 MNIST test images for prediction.

The model predicted the first image as digit **7**.

The actual label was also **7**.

### Prediction Comparison

I used NumPy's `allclose()` function to compare predictions.

```python
np.allclose(p_keras, ref, atol=1e-5)
```

```python
np.allclose(p_saved, ref, atol=1e-5)
```

### Actual Verification Results

```text
Predicted digit: 7 actual: 7

Keras file matches original: True

SavedModel matches original: True
```

**Result:** Both saved models produced predictions matching the original reference predictions within the specified tolerance.

## 11. Important Lab Files

| File | Purpose |
|---|---|
| train.py | Builds and trains the neural network |
| load.py | Loads models and tests predictions |
| requirements.txt | Lists required Python packages |
| model.json | Stores model architecture |
| my_model.keras | Saved Keras model |
| saved_model_dir/ | Exported TensorFlow model |
| ref_preds.npy | Stores reference predictions |

## 12. Important Concepts Learned

**Deep Learning:** A branch of machine learning that uses neural networks to learn patterns from data.

**MNIST:** A handwritten digit image dataset.

**Sequential Model:** A Keras model where layers are arranged one after another.

**Flatten:** Converts multidimensional data into a one-dimensional array.

**Dense Layer:** A neural network layer in which each neuron connects to all neurons in the previous layer.

**ReLU:** An activation function that helps neural networks learn nonlinear patterns.

**Softmax:** Converts output values into probabilities for classification.

**Model Saving:** Stores a trained model so it can be used later.

**Model Exporting:** Packages a model into a format suitable for TensorFlow deployment and serving.

**Inference:** Using a trained model to make predictions.

**Prediction Comparison:** Checking whether two models produce matching outputs.

## 13. Final Verification

- [x] Python virtual environment created.
- [x] TensorFlow CPU installed.
- [x] NumPy installed.
- [x] MNIST dataset downloaded.
- [x] Neural network created.
- [x] Model trained for 2 epochs.
- [x] Keras model saved.
- [x] TensorFlow SavedModel exported.
- [x] Model architecture saved as JSON.
- [x] Reference predictions saved.
- [x] Keras model loaded successfully.
- [x] SavedModel loaded successfully.
- [x] Keras prediction comparison returned True.
- [x] SavedModel prediction comparison returned True.

## 14. Conclusion

In this lab, I successfully trained and deployed a handwritten digit classification model using TensorFlow and Keras.

I learned how to save a trained model in Keras format, export it as a TensorFlow SavedModel, and load both formats separately.

Finally, I verified both models by comparing their predictions against the original reference predictions.

**Lab 35 Status: Completed Successfully.**
