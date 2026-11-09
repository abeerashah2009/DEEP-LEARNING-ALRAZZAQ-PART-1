# Lab 03 – Neural Network Fundamentals

# 1. Introduction to Neural Networks

A Neural Network is a computing model inspired by the human brain. It is made up of interconnected units called neurons.

Neural networks learn patterns from data and use those patterns to produce outputs.

In this lab, we will understand the basic structure of a neural network and implement a simple neural network using Python and NumPy.

### Important Notes

- A neural network contains different layers.
- Each layer contains neurons.
- Neurons receive and process information.
- Weights control the importance of input values.
- Activation functions help neurons produce useful outputs.

# 2. Objectives

After completing this lab, we should be able to:

1. Understand basic neural network architecture.
2. Identify input, hidden, and output layers.
3. Understand neurons and weights.
4. Explain activation functions.
5. Implement Sigmoid, ReLU, and Tanh.
6. Understand the feedforward process.
7. Build a basic neural network using Python and NumPy.

# 3. Prerequisites

Before starting this lab, the following knowledge is useful:

- Basic Python programming.
- Variables and functions.
- Python lists and arrays.
- Basic mathematical operations.
- Linux terminal commands.

# 4. Tools and Technologies

| Tool | Version |
|---|---|
| Operating System | Ubuntu Linux |
| Python | 3.12 |
| NumPy | 1.26.4 |
| Virtual Environment | Python venv |
| Terminal | Linux Terminal |

# 5. Environment Setup

### Step 1: Update the System

Command:

    sudo apt update

Purpose:
Updates the package information available on the Ubuntu system.

### Step 2: Install Python and Virtual Environment

Command:

    sudo apt install -y python3 python3-venv

Purpose:
- python3 installs Python.
- python3-venv provides support for creating virtual environments.

### Step 3: Create a Working Directory

Command:

    mkdir -p ~/nn-lab
    cd ~/nn-lab

Purpose:
Creates a separate directory for the neural network environment.

### Step 4: Create a Virtual Environment

Command:

    python3 -m venv venv

Purpose:
Creates an isolated Python environment for this lab.

### Step 5: Activate the Virtual Environment

Command:

    source venv/bin/activate

Purpose:
Activates the virtual environment.

Note:
When activated, the terminal displays (venv) before the username.

### Step 6: Install NumPy

Command:

    pip install numpy==1.26.4

Purpose:
NumPy is a Python library used for numerical calculations, arrays, and matrix multiplication.

### Step 7: Verify NumPy

Command:

    python3 -c "import numpy; print(numpy.__version__)"

Expected Output:

    1.26.4

Note:
NumPy is required for the neural network calculations in this lab.

# 6. Understanding Neural Network Architecture

A basic neural network consists of three main layers.

### 6.1 Input Layer

The input layer receives the data.

In this practical:
- Number of input features: 3
- Input values: 1, 2, 3

### 6.2 Hidden Layer

The hidden layer processes the input data.

In this practical:
- Number of hidden neurons: 4
- Each neuron receives values from the input layer.
- Weights are used to calculate neuron inputs.
- An activation function is applied.

### 6.3 Output Layer

The output layer produces the final result.

In this practical:
- Number of output neurons: 2
- The final output is calculated using the hidden layer values.

### Architecture Summary

    Input Layer       Hidden Layer       Output Layer
      3 Neurons         4 Neurons          2 Neurons

        [I1]  -------->  [H1]  -------->
        [I2]  -------->  [H2]  -------->  [O1]
        [I3]  -------->  [H3]  -------->
                         [H4]  -------->  [O2]

# 7. Understanding Activation Functions

An activation function determines how a neuron transforms its calculated input.

Three activation functions are implemented in this lab.

### 7.1 Sigmoid Function

Formula:

    sigmoid(x) = 1 / (1 + exp(-x))

Important Notes:
- Converts values into a range between 0 and 1.
- Commonly used for binary classification outputs.
- Helps introduce non-linearity into a neural network.

### 7.2 ReLU Function

Full Name: Rectified Linear Unit

Formula:

    ReLU(x) = max(0, x)

Important Notes:
- Negative values become zero.
- Positive values remain unchanged.
- Commonly used in hidden layers.

### 7.3 Tanh Function

Full Name: Hyperbolic Tangent

Important Notes:
- Converts values into a range between -1 and 1.
- Produces both negative and positive values.
- Its output is centered around zero.

# 8. Practical Implementation

The Python practical is saved in:

    python-practicals/nn.py

### Task 1: Create the Neural Network Diagram

File:

    diagram.txt

Purpose:
Shows the three layers of the neural network and their neuron counts.

### Task 2: Create the Python Program

File:

    python-practicals/nn.py

The implementation includes the following steps:

1. Import NumPy.
2. Create a random number generator with seed 42.
3. Define three input features.
4. Set the hidden layer size to four neurons.
5. Set the output layer size to two neurons.
6. Implement the activation functions.
7. Calculate the hidden layer output.
8. Implement the feedforward process.

### Important NumPy Concepts

**NumPy Array**

An array stores numerical data.

**Weights**

Weights are numerical values used to determine the contribution of inputs to neurons.

**Matrix Multiplication**

The dot product multiplies compatible matrices and arrays.

In this lab, NumPy uses:

    np.dot(input_data, weights)

to calculate the values passed between layers.

# 9. Understanding Feedforward

Feedforward is the process through which information moves from the input layer to the output layer.

### Feedforward Steps

Step 1:
Input data is received by the input layer.

Step 2:
Input values are multiplied by input-to-hidden weights.

Step 3:
The Sigmoid activation function is applied to the hidden layer.

Step 4:
Hidden layer outputs are multiplied by hidden-to-output weights.

Step 5:
Sigmoid activation is applied again to calculate the final output.

### Feedforward Flow

    Input Data
         |
         v
    Input Layer
         |
         v
    Weight Calculation
         |
         v
    Hidden Layer
         |
         v
    Sigmoid Activation
         |
         v
    Output Layer
         |
         v
    Final Output

# 10. Practical Execution

Navigate to the practical directory:

    cd ~/DEEP-LEARNING-ALRAZZAQ-PART-1/Lab-03-Neural-Network-Fundamentals/python-practicals

Run the Python program:

    python3 nn.py

### Actual Practical Results

Input Shape:

    (1, 3)

Hidden Sigmoid Output:

    [[0.79358167 0.97682608 0.97049571 0.99363017]]

ReLU Output:

    [[1.34665164 3.74128096 3.4932714  5.04979161]]

Tanh Output:

    [[0.87326066 0.99887501 0.99815323 0.99991781]]

Feedforward Output:

    [[0.90926386 0.82699475]]

Output Shape:

    (1, 2)

Output Validation:

    True

# 11. Testing and Verification

### Step 1: Verify Feedforward Output

Command:

    python3 nn.py | grep "Feedforward Output"

Expected Output:

    Feedforward Output: [[0.90926386 0.82699475]]

### Step 2: Verify Python Syntax

Command:

    python3 -m py_compile nn.py

Result:
No syntax errors were reported.

### Step 3: Verify Repeatability

The random number generator uses seed 42.

This ensures that the same random weights are generated when the program runs again.

The feedforward output remains the same.

# 12. Important Learning Notes

1. A neural network consists of interconnected neurons.
2. The input layer receives data.
3. The hidden layer processes information.
4. The output layer produces the result.
5. Weights determine the contribution of input values.
6. Activation functions transform neuron inputs.
7. Sigmoid produces values between 0 and 1.
8. ReLU converts negative values to zero.
9. Tanh produces values between -1 and 1.
10. Feedforward moves information from input to output.

# 13. Troubleshooting

### Problem 1: NumPy Module Not Found

Possible Error:

    ModuleNotFoundError: No module named 'numpy'

Solution:

    source ~/nn-lab/venv/bin/activate
    pip install numpy==1.26.4

### Problem 2: Python Syntax Error

Solution:

    python3 -m py_compile nn.py

Check the reported line number and correct the syntax.

### Problem 3: Incorrect Matrix Dimensions

Ensure that the weight dimensions match the input and layer sizes.

For this practical:

    Input-to-hidden weights: (3, 4)
    Hidden-to-output weights: (4, 2)

# 14. Project Structure

    Lab-03-Neural-Network-Fundamentals/
    |
    |-- README.md
    |-- requirements.txt
    |-- diagram.txt
    |
    |-- evidence/
    |
    |-- python-practicals/
        |-- nn.py

# 15. Conclusion

In this lab, a basic neural network was implemented using Python and NumPy.

The practical demonstrated neural network architecture, neurons, weights, activation functions, and feedforward calculations.

The program successfully generated the expected output, passed the syntax check, and produced repeatable results.

This lab provides a foundation for understanding more advanced deep learning concepts.
