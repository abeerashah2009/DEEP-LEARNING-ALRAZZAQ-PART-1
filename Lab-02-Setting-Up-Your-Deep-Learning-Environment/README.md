# Lab 02 – Setting Up Your Deep Learning Environment

## Overview

This lab focuses on preparing a Python environment for Deep Learning development on Ubuntu 24.04.

The environment was configured using Python virtual environments, TensorFlow CPU, NumPy, and Keras. A simple neural network was created and trained to verify that the setup works correctly.

## Objectives

- Set up a Python environment for Deep Learning.
- Understand and use Python virtual environments.
- Install TensorFlow and Keras.
- Install and configure NumPy.
- Verify the installation.
- Build and train a simple neural network.

## Environment Details

| Component | Version / Configuration |
|---|---|
| Operating System | Ubuntu 24.04 LTS |
| Python | 3.12.3 |
| pip | 24.0 |
| TensorFlow CPU | 2.17.0 |
| Keras | 3.15.1 |
| NumPy | 1.26.4 |
| Execution | CPU-only |
| Virtual Environment | my_deeplearning_env |

## Project Structure

```text
Lab-02-Setting-Up-Your-Deep-Learning-Environment/
├── README.md
├── requirements.txt
├── evidence/
└── python-practicals/
    └── test_script.py
```

## Tasks Completed

### 1. Python Environment Setup

Checked the Python version and updated Ubuntu package information.

Installed the required Python tools:

- python3
- python3-venv
- python3-pip

### 2. Virtual Environment Creation

Created a separate Python virtual environment outside the Git repository.

```bash
mkdir -p ~/deeplearning
cd ~/deeplearning
python3 -m venv my_deeplearning_env
```

Verified the Python interpreter:

```bash
my_deeplearning_env/bin/python --version
my_deeplearning_env/bin/python -c 'import sys; print(sys.prefix)'
```

### 3. TensorFlow and NumPy Installation

Installed the required packages using the virtual environment's pip:

```bash
my_deeplearning_env/bin/pip install tensorflow-cpu==2.17.0 numpy==1.26.4
```

### 4. Installation Verification

Verified TensorFlow:

```bash
TF_CPP_MIN_LOG_LEVEL=2 my_deeplearning_env/bin/python -c 'import tensorflow as tf; print(tf.__version__)'
```

Verified TensorFlow and Keras packages:

```bash
my_deeplearning_env/bin/pip list | grep -i -E 'tensorflow|keras'
```

### 5. Neural Network Practical

Created `python-practicals/test_script.py`.

The practical performs the following operations:

1. Imports NumPy, TensorFlow, and Keras.
2. Creates a simple sequential neural network.
3. Configures the model using Stochastic Gradient Descent (SGD).
4. Uses Mean Squared Error as the loss function.
5. Trains the model using sample data for 500 epochs.
6. Predicts the output for an input value of 10.0.
7. Checks whether the prediction falls within the expected range.

Run the practical using:

```bash
TF_CPP_MIN_LOG_LEVEL=2 my_deeplearning_env/bin/python test_script.py
```

## Practical Results

| Test | Result |
|---|---|
| TensorFlow Import | Successful |
| Keras Import | Successful |
| Neural Network Creation | Successful |
| Model Training | Successful |
| Prediction Input | 10.0 |
| Prediction Output | 18.9769 |
| Final Status | PASS |

## Dependencies

The required package versions are recorded in `requirements.txt`.

```text
tensorflow-cpu==2.17.0
numpy==1.26.4
```

## Learning Outcomes

After completing this lab, I gained practical experience in:

- Creating and managing Python virtual environments.
- Installing specific versions of Python libraries.
- Configuring a CPU-based TensorFlow environment.
- Understanding the relationship between TensorFlow and Keras.
- Building and training a basic neural network.
- Testing a deep learning environment through a real practical.

## Conclusion

Lab 02 was completed successfully. The Python virtual environment was created, TensorFlow and NumPy were installed, and Keras was verified.

A simple neural network was trained successfully and generated a prediction of approximately 18.98 for the input value 10.0.

**Final Status: PASSED**
