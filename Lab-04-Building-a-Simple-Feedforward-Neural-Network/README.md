

# Lab 04 – Building a Simple Feedforward Neural Network

## 1. Introduction

A **Feedforward Neural Network (FNN)** is one of the basic types of Artificial Neural Networks (ANNs).

It processes information in one direction:

**Input Layer → Hidden Layers → Output Layer**

Data enters through the input layer, passes through one or more hidden layers, and reaches the output layer. Information does not move backwards during the forward propagation process.

Feedforward Neural Networks are commonly used for classification, prediction, and pattern recognition.

## 2. Lab Objectives

The main objectives of this practical are:

- Understand the basic structure of a feedforward neural network.
- Create a neural network using Python and TensorFlow/Keras.
- Build a Sequential model.
- Add Dense layers with different activation functions.
- Compile the model using an optimizer and loss function.
- Train the model using synthetic data.
- Evaluate model performance and display predictions.
- Understand the model architecture and trainable parameters.

## 3. Tools and Technologies

| Tool | Purpose |
|---|---|
| Python 3.12.3 | Programming language |
| TensorFlow 2.17.0 | Deep learning framework |
| Keras | High-level neural network API |
| NumPy 1.26.4 | Numerical calculations |
| Ubuntu 24.04.3 LTS | Operating system |
| MobaXterm | Remote terminal access |
| Git and GitHub | Version control and portfolio |

## 4. Project Structure

```text
Lab-04-Building-a-Simple-Feedforward-Neural-Network/
│
├── python-practicals/
│   ├── ff_net.py
│   └── requirements.txt
│
├── evidence/
│   └── README.md
│
├── .gitignore
└── README.md
```

## 5. Setting Up the Python Environment

A virtual environment provides an isolated Python environment for installing project dependencies.

### Step 1: Create a working directory

```bash
mkdir -p ~/feedforward-lab
cd ~/feedforward-lab
```

### Step 2: Create a virtual environment

```bash
python3 -m venv venv
```

### Step 3: Activate the environment

```bash
source venv/bin/activate
```

When activated, the terminal displays:

```text
(venv)
```

### Step 4: Install required packages

```bash
pip install tensorflow-cpu==2.17.0 numpy==1.26.4
```

### Step 5: Verify TensorFlow

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

Expected output:

```text
2.17.0
```

## 6. Installing Project Dependencies

The `requirements.txt` file stores the required Python package versions.

Contents:

```text
tensorflow-cpu==2.17.0
numpy==1.26.4
```

To install dependencies from this file:

```bash
pip install -r requirements.txt
```

**Why is this important?**

It helps maintain consistent package versions when recreating the project environment.

## 7. Understanding the Dataset

This practical uses a synthetic dataset generated with NumPy.

The dataset contains:

- 300 samples
- 4 input features per sample
- 3 possible classes
- Random seed: 42

### Input Features

Each sample contains four numerical features.

The model receives these four values as input.

### Output Classes

The model predicts one of three possible classes:

- Class 0
- Class 1
- Class 2

### Generating Synthetic Data

The following code defines three class centres:

```python
centers = np.array([
    [0, 0, 0, 0],
    [3, 3, 0, 0],
    [0, 3, 3, 3]
])
```

Random class labels are generated:

```python
y = rng.integers(0, 3, 300)
```

Input samples are generated around the selected class centres:

```python
X = centers[y] + rng.normal(0, 0.7, (300, 4))
```

The labels are converted into one-hot encoded form:

```python
Y = keras.utils.to_categorical(y, 3)
```

### What is One-Hot Encoding?

One-hot encoding converts class labels into numerical arrays.

Example:

| Class | One-Hot Representation |
|---|---|
| Class 0 | [1, 0, 0] |
| Class 1 | [0, 1, 0] |
| Class 2 | [0, 0, 1] |

This format is suitable for categorical crossentropy with a Softmax output layer.

## 8. Building the Feedforward Neural Network

The model is built using the Keras Sequential API.

### Step 1: Import Libraries

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
```

Explanation:

- `numpy`: Performs numerical operations.
- `tensorflow`: Provides the deep learning framework.
- `keras`: Provides tools for building neural networks.
- `Dense`: Creates fully connected neural network layers.

### Step 2: Set Random Seeds

```python
np.random.seed(42)
tf.random.set_seed(42)
```

Random seeds help make random operations more reproducible.

### Step 3: Initialize the Sequential Model

```python
model = keras.Sequential([
    keras.Input(shape=(4,))
])
```

The Sequential model allows layers to be added one after another.

`shape=(4,)` means the model accepts four input features per sample.

### Step 4: Add the First Hidden Layer

```python
model.add(Dense(units=8, activation='relu'))
```

This layer contains eight neurons.

**ReLU (Rectified Linear Unit)** is an activation function that returns zero for negative inputs and the input value for positive inputs.

### Step 5: Add the Second Hidden Layer

```python
model.add(Dense(units=16, activation='relu'))
```

This layer contains sixteen neurons.

It receives information from the first hidden layer and learns more complex patterns.

### Step 6: Add the Output Layer

```python
model.add(Dense(units=3, activation='softmax'))
```

The output layer contains three neurons because the dataset has three classes.

**Softmax** converts output values into probabilities that sum to approximately 1.

The class with the highest probability is the predicted class.

## 9. Understanding the Model Architecture

The network structure is:

```text
       INPUT LAYER
       4 Features
           |
           v
    HIDDEN LAYER 1
       8 Neurons
       ReLU
           |
           v
    HIDDEN LAYER 2
       16 Neurons
       ReLU
           |
           v
      OUTPUT LAYER
       3 Neurons
       Softmax
           |
           v
    Predicted Class
```

### Layer Summary

| Layer | Neurons | Activation |
|---|---:|---|
| Input | 4 features | None |
| Hidden Layer 1 | 8 | ReLU |
| Hidden Layer 2 | 16 | ReLU |
| Output Layer | 3 | Softmax |

## 10. Compiling the Model

Before training, the model must be compiled.

```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

### Understanding Compilation Settings

**Optimizer: Adam**

Adam is an optimization algorithm that updates the model's weights during training to reduce errors.

**Loss Function: Categorical Crossentropy**

This loss function measures the difference between the actual one-hot encoded class labels and the predicted class probabilities.

**Metric: Accuracy**

Accuracy measures the proportion of correct predictions.

## 11. Displaying the Model Summary

```python
model.summary()
```

This command displays the neural network architecture, output shapes, and trainable parameters.

### Actual Model Summary

| Layer | Output Shape | Parameters |
|---|---|---:|
| Dense | (None, 8) | 40 |
| Dense | (None, 16) | 144 |
| Dense | (None, 3) | 51 |
| **Total** | | **235** |

### Understanding Parameter Calculation

Each Dense layer has weights and biases.

Formula:

**Parameters = (Input Neurons × Output Neurons) + Output Neurons**

First Dense layer:

```text
(4 × 8) + 8 = 40
```

Second Dense layer:

```text
(8 × 16) + 16 = 144
```

Output Dense layer:

```text
(16 × 3) + 3 = 51
```

Total parameters:

```text
40 + 144 + 51 = 235
```

Therefore:

- Total parameters: 235
- Trainable parameters: 235
- Non-trainable parameters: 0

## 12. Training the Model

The model is trained using:

```python
model.fit(X, Y, epochs=50, batch_size=16, verbose=0)
```

### Understanding Training Settings

**Epochs = 50**

An epoch means one complete pass through the training dataset.

The model processes the training dataset 50 times.

**Batch Size = 16**

The model processes 16 samples in each batch before updating its weights.

**Verbose = 0**

This disables the training progress display.

## 13. Evaluating Model Performance

After training, the model is evaluated using:

```python
loss, acc = model.evaluate(X, Y, verbose=0)
```

The final accuracy is displayed with:

```python
print(f"Final accuracy: {acc:.3f}")
```

The model achieved:

```text
Final accuracy: 0.997
```

This is approximately **99.3% training accuracy**.

Note: Evaluation uses the same synthetic dataset used for training. Therefore, this result does not represent performance on independent test data.

## 14. Making Predictions

The first three samples are passed to the model:

```python
print("True classes of first 3 samples:", y[:3])

print(np.round(model.predict(X[:3], verbose=0), 3))
```

Actual true classes:

```text
[0 2 1]
```

Actual model predictions:

```text
[[0.99  0.008 0.001]
 [0.    0.    1.   ]
 [0.    1.    0.   ]]
```

### Understanding the Predictions

Each row represents the predicted probabilities for three classes.

- Sample 1: Class 0 has the highest probability.
- Sample 2: Class 2 has the highest probability.
- Sample 3: Class 1 has the highest probability.

The predictions match the true classes of all three samples.

## 15. Important Deep Learning Concepts

| Concept | Explanation |
|---|---|
| Neural Network | A model inspired by interconnected neurons |
| Feedforward | Information moves from input to output |
| Sequential Model | A model made of layers arranged in order |
| Dense Layer | A fully connected neural network layer |
| Neuron | A processing unit in a neural network |
| ReLU | Activation function used in hidden layers |
| Softmax | Converts output scores into class probabilities |
| Optimizer | Updates model weights during training |
| Loss Function | Measures prediction error |
| Epoch | One complete pass through training data |
| Batch Size | Number of samples processed in one batch |
| Accuracy | Proportion of correct predictions |
| One-Hot Encoding | Converts class labels into binary vectors |

## 16. Running the Practical

Activate the virtual environment:

```bash
source ~/feedforward-lab/venv/bin/activate
```

Move to the Lab 04 folder:

```bash
cd ~/DEEP-LEARNING-ALRAZZAQ-PART-1/Lab-04-Building-a-Simple-Feedforward-Neural-Network
```

Run the practical:

```bash
python python-practicals/ff_net.py
```

## 17. Troubleshooting

### Problem 1: TensorFlow Module Not Found

Error:

```text
ModuleNotFoundError: No module named 'tensorflow'
```

Solution:

Activate the virtual environment and install TensorFlow:

```bash
source ~/feedforward-lab/venv/bin/activate
pip install tensorflow-cpu==2.17.0
```

### Problem 2: NumPy Module Not Found

Error:

```text
ModuleNotFoundError: No module named 'numpy'
```

Solution:

```bash
pip install numpy==1.26.4
```

### Problem 3: Virtual Environment Not Active

Check the terminal prompt.

If `(venv)` is missing, activate it:

```bash
source ~/feedforward-lab/venv/bin/activate
```

### Problem 4: Check Python Version

```bash
python3 --version
```

Expected:

```text
Python 3.12.3
```

## 18. Interview Questions and Answers

**Q1. What is a feedforward neural network?**

A feedforward neural network is a neural network in which information moves from the input layer through hidden layers to the output layer.

**Q2. What is a Dense layer?**

A Dense layer is a fully connected layer in which every neuron receives input from all neurons in the previous layer.

**Q3. Why is ReLU used in hidden layers?**

ReLU introduces non-linearity and helps the network learn complex patterns.

**Q4. Why is Softmax used in the output layer?**

Softmax converts output scores into probabilities for classification.

**Q5. What is the purpose of an optimizer?**

An optimizer updates the model weights to reduce the loss during training.

**Q6. What is an epoch?**

An epoch is one complete pass through the training dataset.

**Q7. What is batch size?**

Batch size is the number of samples processed together before a weight update.

**Q8. Why do we use categorical crossentropy?**

It measures the prediction error for multi-class classification when labels are one-hot encoded.

**Q9. How many trainable parameters are in this model?**

There are 235 trainable parameters.

**Q10. What is the purpose of model.summary()?**

It displays the model layers, output shapes, and parameter counts.

## 19. Final Lab Results

| Task | Status |
|---|---|
| Python environment setup | Completed |
| TensorFlow and NumPy installation | Completed |
| Synthetic dataset generation | Completed |
| Sequential model creation | Completed |
| Dense layer implementation | Completed |
| Model compilation | Completed |
| Model summary | Completed |
| Model training | Completed |
| Model evaluation | Completed |
| Sample predictions | Completed |

### Final Output

- Model: Sequential Feedforward Neural Network
- Input features: 4
- Hidden layers: 2
- Output classes: 3
- Total parameters: 235
- Epochs: 50
- Batch size: 16
- Training accuracy: 99.3%
- First three predictions: Correct

## 20. Conclusion

In this practical, I successfully built and trained a simple feedforward neural network using Python, TensorFlow, Keras, and NumPy.

I learned how to generate synthetic data, create a Sequential model, add Dense layers, select activation functions, compile the model, train it, and evaluate its predictions.

The model achieved 99.3% accuracy on the synthetic training dataset and correctly classified the first three samples.

This practical strengthened my understanding of neural network architecture and the basic workflow used in deep learning projects.

---

**Lab 04: Building a Simple Feedforward Neural Network**

**Status: Completed**
