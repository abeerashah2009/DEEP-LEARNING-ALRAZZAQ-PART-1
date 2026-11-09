# Lab 03 - Evidence and Verification

This folder is used to organize evidence for Lab 03: Neural Network Fundamentals.

## Evidence Checklist

### 1. Python Environment
Capture the terminal showing:
- Python version
- Virtual environment activated
- NumPy version 1.26.4

### 2. Neural Network Architecture
Capture:
- The contents of `diagram.txt`
- Input layer with 3 nodes
- Hidden layer with 4 neurons
- Output layer with 2 neurons

### 3. Neural Network Implementation
Capture the `nn.py` code showing:
- NumPy import
- Input data
- Layer sizes
- Weight matrices
- Activation functions
- Feedforward function

### 4. Program Execution
Capture the complete output showing:
- Input shape
- Sigmoid output
- ReLU output
- Tanh output
- Feedforward output
- Output shape
- Value range verification

### 5. Testing and Verification
Capture:
- Successful Python execution
- Deterministic feedforward output
- Python syntax compilation without errors

## Verification Commands

Run these commands from the Lab 03 directory:

```bash
python3 --version
python3 -c "import numpy; print(numpy.__version__)"
cat diagram.txt
cat python-practicals/nn.py
python3 python-practicals/nn.py
python3 -m py_compile python-practicals/nn.py
python3 python-practicals/nn.py | grep "Feedforward Output"

```
