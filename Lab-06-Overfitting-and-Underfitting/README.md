# Lab 06 – Overfitting and Underfitting

## 1. Introduction

In this lab, I explored two important problems in Machine Learning and Deep Learning: **Overfitting and Underfitting**.

I used Python, NumPy, Matplotlib, and Scikit-learn to train polynomial regression models, compare training and validation errors, visualize model performance, and explore regularization techniques.

The practical also introduced Dropout and Ridge Regularization as methods to reduce overfitting.

## 2. Learning Objectives

By completing this lab, I learned how to:

- Understand overfitting and underfitting.
- Identify overfitting using training and validation loss.
- Understand the relationship between model complexity and performance.
- Prepare training and validation datasets.
- Train polynomial regression models.
- Compare different polynomial degrees.
- Visualize training and validation errors.
- Understand the concept of Dropout.
- Explore Ridge Regularization.
- Evaluate models using Mean Squared Error (MSE).

## 3. Understanding Overfitting

### What is Overfitting?

Overfitting happens when a machine learning model learns the training data too closely, including noise and unnecessary details.

As a result, the model performs very well on training data but performs poorly on new, unseen data.

**Example:**

A student memorizes every question from a practice test but cannot answer different questions in the actual examination.

Similarly, an overfitted model memorizes training patterns instead of learning general patterns.

### Signs of Overfitting

- Very low training error.
- Very high validation error.
- Large difference between training and validation performance.
- Poor performance on unseen data.

### Common Causes

- Model is too complex.
- Training dataset is too small.
- Too many model parameters.
- Model learns noise in the dataset.
- Insufficient regularization.

## 4. Understanding Underfitting

### What is Underfitting?

Underfitting happens when a model is too simple to learn the important patterns in the training data.

It performs poorly on both training and validation datasets.

**Example:**

A student studies only one topic for an examination and cannot answer questions from other topics.

### Signs of Underfitting

- High training error.
- High validation error.
- Model does not learn important patterns.
- Model complexity is insufficient.

### Common Causes

- Model is too simple.
- Important features are missing.
- Training is insufficient.
- Model has excessive regularization.

## 5. Overfitting vs Underfitting

| Feature | Underfitting | Overfitting |
|---|---|---|
| Model complexity | Too low | Too high |
| Training error | High | Low |
| Validation error | High | Very high |
| Learning | Does not learn enough | Learns training data too closely |
| Generalization | Poor | Poor |
| Possible solution | Increase model complexity | Regularization or simpler model |

## 6. Training Loss and Validation Loss

### Training Loss

Training loss measures how much error a model makes on the data used to train it.

A lower training loss generally means the model fits its training dataset better.

### Validation Loss

Validation loss measures model performance on separate data that was not used for training.

It helps us understand whether the model can generalize to unseen examples.

### Why Compare Both?

Comparing training and validation loss helps identify model behaviour.

- Both losses high: possible underfitting.
- Training loss low and validation loss high: possible overfitting.
- Both losses low and reasonably close: good generalization.

## 7. Mean Squared Error (MSE)

MSE is an evaluation metric used to measure the average squared difference between actual and predicted values.

**Formula:**

MSE = (1/n) × Σ(actual value − predicted value)²

Where:

- n = Number of samples.
- Actual value = Correct target value.
- Predicted value = Model prediction.
- Squared difference = Error after squaring.

A smaller MSE indicates better predictions on the evaluated dataset.

## 8. Bias and Variance

### Bias

Bias is the error caused by assumptions that are too simple for the actual data.

High bias can lead to underfitting.

### Variance

Variance describes how sensitive a model is to changes in training data.

High variance can lead to overfitting.

### Bias-Variance Tradeoff

The goal is to balance bias and variance so the model performs well on both training and unseen data.

- High bias: model is too simple.
- High variance: model is too sensitive to training data.
- Balanced model: better generalization.

## 9. Lab Environment Setup

**Operating System:** Ubuntu 24.04 LTS

**Python Version:** 3.12

**Libraries Used:**

| Library | Version | Purpose |
|---|---|---|
| scikit-learn | 1.5.2 | Machine Learning |
| NumPy | 1.26.4 | Numerical calculations |
| Matplotlib | 3.9.2 | Graph visualization |

### Step 1: Update Ubuntu

```bash
sudo apt update
```

Purpose: Updates the package information.

### Step 2: Install Python and Virtual Environment

```bash
sudo apt install -y python3 python3-venv
```

Purpose: Installs Python and support for isolated virtual environments.

### Step 3: Create a Working Directory

```bash
mkdir -p ~/overfit-lab
cd ~/overfit-lab
```

Purpose: Creates a separate environment directory.

### Step 4: Create a Virtual Environment

```bash
python3 -m venv .venv
```

Purpose: Creates an isolated Python environment.

### Step 5: Activate the Environment

```bash
source .venv/bin/activate
```

Purpose: Activates the virtual environment.

### Step 6: Install Required Libraries

```bash
pip install scikit-learn==1.5.2 numpy==1.26.4 matplotlib==3.9.2
```

Purpose: Installs the required libraries with fixed versions.

### Step 7: Verify Installation

```bash
python3 -c "import sklearn, numpy, matplotlib; print(sklearn.__version__); print(numpy.__version__); print(matplotlib.__version__)"
```

Verified versions:

- Scikit-learn: 1.5.2
- NumPy: 1.26.4
- Matplotlib: 3.9.2

## 10. Dataset Preparation

For this practical, I used the Diabetes dataset available in Scikit-learn.

The dataset contains numerical information used for a regression task.

I selected:

- 50 random samples.
- BMI as the input feature.
- Diabetes target values as the output.

The dataset was divided into:

- 25 training samples.
- 25 validation samples.

### Python File: data.py

Location: `python-practicals/data.py`

The script performs the following operations:

1. Imports NumPy.
2. Loads the Diabetes dataset.
3. Selects 50 random samples.
4. Extracts the BMI feature.
5. Separates training and validation data.

**Verification Command:**

```bash
python3 -c "from data import *; print(X_train.shape, X_val.shape)"
```

**Output:**

```text
(25, 1) (25, 1)
```

## 11. Polynomial Regression

Polynomial Regression is a regression technique that models relationships using polynomial features.

For example:

- Degree 1: Linear relationship.
- Degree 2: Quadratic relationship.
- Degree 3: Cubic relationship.
- Degree 10: More complex polynomial relationship.

Increasing polynomial degree increases model flexibility and can also increase the risk of overfitting.

## 12. Simple Model Comparison

### Practical File: simple.py

Location: `python-practicals/simple.py`

In this experiment, I compared two models:

- Degree 1 polynomial regression.
- Degree 10 polynomial regression.

Both models used the same training and validation datasets.

### Actual Results

| Degree | Training MSE | Validation MSE |
|---|---:|---:|
| 1 | 3994.7 | 4602.9 |
| 10 | 1791.6 | 4139436675908272.0 |

### Interpretation

**Degree 1:**

The simple model had relatively high training and validation errors.

This suggests that the model may not capture all the patterns in the data.

**Degree 10:**

The complex model reduced training error but produced an extremely high validation error.

This is a strong example of overfitting.

## 13. Polynomial Degree Sweep

### Practical File: sweep.py

Location: `python-practicals/sweep.py`

In this experiment, I tested polynomial degrees from 1 to 10.

For each degree, I:

1. Created a polynomial regression pipeline.
2. Trained the model.
3. Calculated training MSE.
4. Calculated validation MSE.
5. Calculated the difference between both errors.
6. Saved the results in a text file.

### Results

| Degree | Training MSE | Validation MSE |
|---:|---:|---:|
| 1 | 3994.7 | 4602.9 |
| 2 | 3985.3 | 4159.9 |
| 3 | 3947.8 | 6937.7 |
| 4 | 3923.9 | 89840.2 |
| 5 | 3527.4 | 30507516.1 |
| 6 | 2811.7 | 1419496583.7 |
| 7 | 2591.7 | 33599705096.3 |
| 8 | 2572.1 | 200215601815.5 |
| 9 | 2524.1 | 6267369589376.8 |
| 10 | 1791.6 | 4139436675908272.0 |

**Best Degree:** 2

The degree-2 model achieved the lowest validation MSE of 4159.9.

### Important Observation

As polynomial degree increased:

- Training error generally decreased.
- Validation error increased dramatically.
- The gap between training and validation error became much larger.

This demonstrates how an overly complex model can overfit the training data.

## 14. Training vs Validation Loss Graph

The practical generated a graph named:

`loss.png`

The graph displays:

- Polynomial degree on the X-axis.
- Mean Squared Error on the Y-axis.
- Training MSE.
- Validation MSE.

A logarithmic scale was used for the Y-axis because validation errors became extremely large.

**Purpose of the Graph:**

It helps visualize the relationship between model complexity and generalization performance.

The widening gap between training and validation errors indicates overfitting.

## 15. Dropout Regularization

### What is Dropout?

Dropout is a regularization technique used in neural networks.

During training, dropout randomly disables selected neurons.

This prevents the network from depending too heavily on particular neurons and can improve generalization.

### Dropout Probability

In this practical, the dropout probability was:

```python
p = 0.5
```

This means each neuron has a 50% probability of being disabled during the demonstration.

### Practical Demonstration

A small NumPy example was used to demonstrate the dropout mechanism.

**Output:**

```text
[2. 0. 2. 2. 0. 2. 2. 2. 0. 0.]
```

Interpretation:

- Zero represents a disabled neuron.
- Two represents an active neuron after scaling.

The active values are scaled by:

1 / (1 - p)

This maintains the expected activation magnitude.

**Important:** This was a mathematical illustration of dropout, not an actual trained neural network.

## 16. Ridge Regularization

### What is Ridge Regularization?

Ridge Regression is a regularization technique that adds a penalty to large model coefficients.

It helps control model complexity and can reduce overfitting.

Ridge uses L2 regularization.

### Alpha Parameter

The alpha parameter controls the strength of regularization.

- Small alpha: weaker regularization.
- Large alpha: stronger regularization.

An excessively large alpha can also lead to underfitting.

### Practical File: ridge.py

Location: `python-practicals/ridge.py`

I compared:

1. Plain Linear Regression.
2. Ridge Regression with alpha = 10.0.

### Actual Results

| Model | Validation MSE |
|---|---:|
| Plain Linear Regression | 4139436675908272.0 |
| Ridge (alpha = 10.0) | 20969454.8 |

### Interpretation

Ridge regularization significantly reduced validation error compared with plain regression.

However, the Ridge model still had a higher validation error than the degree-2 polynomial model.

This shows that selecting an appropriate model complexity is also important.

## 17. Techniques to Reduce Overfitting

Common methods include:

### 1. Dropout

Randomly disables neurons during neural network training.

### 2. Ridge Regularization

Penalizes large coefficients using L2 regularization.

### 3. Lasso Regularization

Uses L1 regularization and can reduce some coefficients to zero.

### 4. More Training Data

Additional representative data can help a model learn general patterns.

### 5. Simpler Model

Reducing unnecessary model complexity can improve generalization.

### 6. Early Stopping

Stops training when validation performance stops improving.

### 7. Cross-Validation

Evaluates a model across multiple training and validation splits.

## 18. Project Directory Structure

```text
Lab-06-Overfitting-and-Underfitting/
│
├── python-practicals/
│   ├── data.py
│   ├── simple.py
│   ├── sweep.py
│   ├── ridge.py
│   └── table.txt
│
├── evidence/
│   └── README.md
│
├── loss.png
└── README.md
```

## 19. Important Commands Used

| Command | Purpose |
|---|---|
| `sudo apt update` | Updates package information |
| `python3 -m venv .venv` | Creates virtual environment |
| `source .venv/bin/activate` | Activates environment |
| `pip install` | Installs Python libraries |
| `python3 data.py` | Runs dataset preparation |
| `python3 simple.py` | Compares two models |
| `python3 sweep.py` | Tests polynomial degrees |
| `python3 ridge.py` | Tests Ridge regularization |
| `ls -lh loss.png` | Checks graph file |
| `file loss.png` | Checks image type and dimensions |
| `cat table.txt` | Displays model results |

## 20. Interview Questions and Answers

### Q1. What is overfitting?

Overfitting occurs when a model learns training data too closely and performs poorly on unseen data.

### Q2. What is underfitting?

Underfitting occurs when a model is too simple to learn important patterns in the dataset.

### Q3. What is training loss?

Training loss measures the model's error on training data.

### Q4. What is validation loss?

Validation loss measures the model's performance on data not used during training.

### Q5. What is MSE?

Mean Squared Error measures the average squared difference between actual and predicted values.

### Q6. What is Dropout?

Dropout randomly disables selected neurons during neural network training to reduce overfitting.

### Q7. What is Ridge Regression?

Ridge Regression is a regularized regression technique that uses an L2 penalty to control large coefficients.

### Q8. What is the difference between L1 and L2 regularization?

L1 regularization can reduce coefficients to zero, while L2 regularization penalizes large coefficients.

### Q9. What is the bias-variance tradeoff?

It is the balance between errors caused by overly simple assumptions and errors caused by sensitivity to training data.

### Q10. Which polynomial degree performed best in this experiment?

Degree 2 achieved the lowest validation MSE of 4159.9.

### Q11. Why did degree 10 overfit?

The degree-10 model was too complex for the small dataset and produced a very large validation error.

### Q12. Why do we use a validation dataset?

A validation dataset helps evaluate how well a model performs on data not used for training.

## 21. Troubleshooting

### Problem 1: ModuleNotFoundError

**Cause:** Python could not find the module.

**Solution:** Run the practical from the `python-practicals` directory when importing `data.py`.

### Problem 2: pip command not found

**Cause:** The virtual environment may not be activated or the command may have been pasted incorrectly.

**Solution:**

```bash
source ~/overfit-lab/.venv/bin/activate
python3 -m pip --version
```

### Problem 3: Graph not generated

**Cause:** Matplotlib may not be installed or the script may have failed.

**Solution:**

```bash
pip install matplotlib==3.9.2
```

Then run:

```bash
MPLBACKEND=Agg python3 sweep.py
```

### Problem 4: Very Large Validation Error

A very large validation error can occur with a high-degree polynomial model.

In this lab, it was an expected observation demonstrating overfitting rather than an installation failure.

## 22. Final Conclusion

In this lab, I learned how overfitting and underfitting affect machine learning model performance.

I prepared a dataset, trained polynomial regression models, compared training and validation errors, generated a loss graph, demonstrated dropout, and explored Ridge regularization.

The experiments showed that increasing model complexity can reduce training error while increasing validation error.

The degree-2 model achieved the best validation result in this experiment. Ridge regularization also significantly reduced the validation error of the degree-10 model.

**Key Learning:** A good machine learning model should not simply memorize training data. It should learn useful patterns and perform well on unseen data.

**Lab 06 – Overfitting and Underfitting: Completed.**
