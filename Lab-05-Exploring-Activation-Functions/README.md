# Lab 05 – Exploring Activation Functions

## 1. Introduction

Activation functions are important components of artificial neural networks. They help neural networks learn complex patterns by introducing non-linearity into the model.

Without activation functions, a neural network would behave like a linear model, even if it contained multiple layers.

In this practical, three activation functions are explored:

1. ReLU (Rectified Linear Unit)
2. Sigmoid
3. Tanh (Hyperbolic Tangent)

The experiment compares their training behavior and loss values.

## 2. Learning Objectives

After completing this practical, I learned how to:

- Understand the purpose of activation functions.
- Implement ReLU, Sigmoid, and Tanh in a neural network.
- Build neural network models using Keras.
- Train models with different activation functions.
- Compare training loss values.
- Visualize training performance using Matplotlib.
- Identify which activation function achieved the lowest final loss in this experiment.

## 3. Tools and Technologies

| Tool | Version |
|---|---|
| Python | 3.12.3 |
| TensorFlow CPU | 2.17.0 |
| Keras | Installed with TensorFlow |
| NumPy | 1.26.4 |
| Matplotlib | 3.9.2 |
| Operating System | Ubuntu 24.04 LTS |

## 4. Understanding Activation Functions

An activation function determines the output of a neuron based on its input.

It helps a neural network learn non-linear relationships in data.

### 4.1 ReLU

ReLU stands for Rectified Linear Unit.

Formula:

`f(x) = max(0, x)`

Explanation:

- If the input is positive, ReLU returns the same value.
- If the input is negative, ReLU returns zero.
- It is computationally simple.
- It is widely used in hidden layers.

Advantages:
- Simple and fast.
- Helps reduce the vanishing gradient problem for positive inputs.

Disadvantage:
- Some neurons may stop activating. This is known as the Dying ReLU problem.

### 4.2 Sigmoid

Sigmoid converts an input into a value between 0 and 1.

Formula:

`f(x) = 1 / (1 + e^(-x))`

Explanation:

- Output is always between 0 and 1.
- Often useful in the output layer of binary classification models.
- Produces smooth output values.

Advantages:
- Easy to interpret as a probability-like output.
- Useful for binary classification output layers.

Disadvantages:
- Can suffer from vanishing gradients.
- May learn more slowly in deep networks.

### 4.3 Tanh

Tanh stands for Hyperbolic Tangent.

Formula:

`f(x) = tanh(x)`

Explanation:

- Output ranges between -1 and +1.
- Its output is zero-centered.
- It is commonly used in neural network hidden layers.

Advantages:
- Zero-centered output.
- Can perform better than Sigmoid in some hidden-layer situations.

Disadvantage:
- Can also suffer from vanishing gradients for very large positive or negative inputs.

## 5. Practical Environment Setup

### Step 1: Update Ubuntu

```bash
sudo apt update
```

### Step 2: Install Python virtual environment support

```bash
sudo apt install -y python3.12-venv
```

### Step 3: Create a separate virtual environment

```bash
mkdir -p ~/activation-lab
cd ~/activation-lab
python3 -m venv venv
```

### Step 4: Activate the environment

```bash
source venv/bin/activate
```

### Step 5: Install required libraries

```bash
pip install tensorflow-cpu==2.17.0 matplotlib==3.9.2 "numpy<2.2"
```

### Step 6: Verify installed versions

```bash
python3 -c "import tensorflow as tf, matplotlib, numpy; print(tf.__version__, matplotlib.__version__, numpy.__version__)"
```

Expected versions:

- TensorFlow: 2.17.0
- Matplotlib: 3.9.2
- NumPy: 1.26.4

Note: The virtual environment is stored outside this GitHub project folder.

## 6. Understanding the Dataset

The experiment uses synthetic training data created with NumPy.

```python
X_train = np.random.rand(500, 10)
y_train = np.sin(2 * np.pi * X_train).sum(axis=1, keepdims=True)
```

Explanation:

- `500` represents the number of training samples.
- `10` represents the number of input features in each sample.
- `X_train` contains randomly generated input values.
- `y_train` contains target values calculated using a sine function.

The neural network learns to predict the target values from the input features.

## 7. Neural Network Architecture

The experiment uses a simple feedforward neural network.

Architecture:

- Input layer: 10 input features.
- Hidden layer: 64 neurons.
- Hidden activation: ReLU, Sigmoid, or Tanh.
- Output layer: 1 neuron.

The output layer uses a linear activation because this is a regression problem.

### Model Compilation

```python
model.compile(
    optimizer="adam",
    loss="mse"
)
```

Explanation:

**Adam Optimizer:** Updates the model weights during training.

**Mean Squared Error (MSE):** Measures the average squared difference between predicted and actual values.

Lower MSE indicates smaller prediction errors on the evaluated dataset.

## 8. Training Configuration

Each model is trained using the following settings:

| Parameter | Value |
|---|---|
| Activation functions | ReLU, Sigmoid, Tanh |
| Epochs | 30 |
| Batch size | 32 |
| Optimizer | Adam |
| Loss function | MSE |
| Hidden neurons | 64 |

A separate model is created for each activation function.

## 9. Training Results

The final recorded experiment produced these results:

| Activation | First Loss | Epoch 5 Loss | Final Loss |
|---|---:|---:|---:|
| ReLU | 4.7862 | 4.5049 | 2.3620 |
| Sigmoid | 4.8492 | 4.5965 | 3.2848 |
| Tanh | 4.8054 | 4.5067 | 2.3355 |

### Result Analysis

**ReLU:**
- Reduced the training loss from 4.7862 to 2.3620.
- Showed a substantial reduction in loss during training.

**Sigmoid:**
- Reduced the training loss from 4.8492 to 3.2848.
- Had the highest final loss in this experiment.

**Tanh:**
- Reduced the training loss from 4.8054 to 2.3355.
- Achieved the lowest final loss among the three models.

### Best Result

Tanh achieved the lowest final training loss of 2.3355.

This result applies to this particular dataset, architecture, and training run. It does not mean Tanh is always the best activation function.

Note: These values are training losses, not test accuracy. Results may vary slightly between training runs.

## 10. Loss Visualization

Matplotlib is used to compare the loss values across training epochs.

The practical generates:

`loss.png`

The graph contains:

- X-axis: Epoch
- Y-axis: Loss (MSE)
- Three lines: ReLU, Sigmoid, and Tanh

The graph helps visualize how the training loss changes during model training.

## 11. Important Python Concepts Used

### NumPy

Used to generate synthetic training data and calculate target values.

### TensorFlow and Keras

Used to create, compile, and train neural network models.

### Matplotlib

Used to visualize training loss.

### Dictionaries

The `histories` dictionary stores the training loss for each activation function.

### Loops

A `for` loop trains the models using all three activation functions.

## 12. How to Run the Practical

Navigate to the Lab 05 directory:

```bash
cd ~/DEEP-LEARNING-ALRAZZAQ-PART-1/Lab-05-Exploring-Activation-Functions
```

Activate the environment:

```bash
source ~/activation-lab/venv/bin/activate
```

Install dependencies if needed:

```bash
pip install -r python-practicals/requirements.txt
```

Run the practical:

```bash
python3 python-practicals/activations.py
```

The program displays the training results and generates `loss.png`.

## 13. Project Structure

```text
Lab-05-Exploring-Activation-Functions/
│
├── python-practicals/
│   ├── activations.py
│   └── requirements.txt
│
├── evidence/
│   ├── README.md
│   └── training-results.txt
│
├── loss.png
└── README.md
```

## 14. Interview Questions and Answers

**Q1. What is an activation function?**

An activation function determines the output of a neuron and introduces non-linearity into a neural network.

**Q2. Why do neural networks need activation functions?**

They allow neural networks to learn complex, non-linear patterns.

**Q3. What is ReLU?**

ReLU stands for Rectified Linear Unit. It returns zero for negative inputs and the input itself for positive values.

**Q4. What is Sigmoid?**

Sigmoid is an activation function that produces values between 0 and 1.

**Q5. What is Tanh?**

Tanh is an activation function that produces values between -1 and +1.

**Q6. What is the difference between Sigmoid and Tanh?**

Sigmoid produces values between 0 and 1, while Tanh produces values between -1 and +1.

**Q7. What is the vanishing gradient problem?**

It occurs when gradients become extremely small during backpropagation, making it difficult for a neural network to learn.

**Q8. What is an epoch?**

An epoch represents one complete pass through the training dataset.

**Q9. What is batch size?**

Batch size is the number of training samples processed before the model updates its weights.

**Q10. What is MSE?**

Mean Squared Error measures the average squared difference between predicted and actual values.

**Q11. Which activation function achieved the lowest final loss in this practical?**

Tanh achieved the lowest final loss of 2.3355.

**Q12. Is Tanh always better than ReLU?**

No. The best activation function depends on the dataset,
