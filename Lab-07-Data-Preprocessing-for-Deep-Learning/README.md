# Lab 07 – Data Preprocessing for Deep Learning

## 1. Lab Overview

Data preprocessing is an important step in Deep Learning and Machine Learning. Before training a model, raw data must be prepared in a suitable format.

In this lab, I learned how to normalize and standardize numerical data, divide a dataset into training and testing sets, and apply scaling without causing data leakage.

## 2. Learning Objectives

By completing this lab, I learned how to:

- Understand the importance of data preprocessing.
- Apply MinMaxScaler for normalization.
- Apply StandardScaler for standardization.
- Create numerical datasets using NumPy.
- Split data into training and testing sets.
- Use a reproducible train/test split.
- Prevent data leakage during preprocessing.
- Verify preprocessing results using Python.

## 3. Environment Setup

### Operating System
- Ubuntu 24.04 LTS

### Programming Language
- Python 3.12.3

### Python Libraries

| Library | Version | Purpose |
|---|---|---|
| NumPy | 1.26.4 | Numerical calculations and arrays |
| Scikit-learn | 1.5.2 | Data preprocessing and dataset splitting |
| SciPy | 1.17.1 | Scientific computing dependency |

### Create a Virtual Environment

```bash
mkdir -p ~/preprocess-lab
cd ~/preprocess-lab
python3 -m venv venv
source venv/bin/activate
```

### Install Required Libraries

```bash
pip install numpy==1.26.4 scikit-learn==1.5.2
```

### Verify Installed Versions

```bash
python3 -c "import numpy, sklearn; print(numpy.__version__, sklearn.__version__)"
```

Expected output:

```text
1.26.4 1.5.2
```

## 4. What Is Data Preprocessing?

Data preprocessing is the process of preparing raw data before using it to train a Machine Learning or Deep Learning model.

Raw data may contain values with different ranges and scales. These differences can affect model training.

Preprocessing helps make data more consistent and suitable for learning algorithms.

### Why Is Data Preprocessing Important?

- Improves data consistency.
- Helps numerical features use suitable scales.
- Supports more stable model training.
- Makes datasets ready for Machine Learning algorithms.
- Helps prevent data leakage when performed correctly.

## 5. Normalization – MinMaxScaler

Normalization is a scaling technique that transforms numerical features into a specified range.

In this lab, I used MinMaxScaler with its default range of 0 to 1.

### Python Example

```python
from sklearn.preprocessing import MinMaxScaler
import numpy as np

data = np.array([
    [10, 2.7, 3.6],
    [-100, 5, -2],
    [120, 20, 40]
], dtype=np.float64)

normalized_sample = MinMaxScaler().fit_transform(data)

print(normalized_sample.round(3))
```

### Explanation

- `MinMaxScaler()` creates the normalization scaler.
- `fit_transform()` learns the minimum and maximum values from the supplied data and transforms it.
- `round(3)` displays results to three decimal places.

### Actual Output

```text
[[0.5   0.    0.133]
 [0.    0.133 0.   ]
 [1.    1.    1.   ]]
```

### Key Point

Each feature column is scaled independently.

## 6. Standardization – StandardScaler

Standardization transforms numerical features so that each feature has a mean close to 0 and a standard deviation close to 1.

### Python Example

```python
from sklearn.preprocessing import StandardScaler

standardized_sample = StandardScaler().fit_transform(data)

print(standardized_sample.round(3))
print(standardized_sample.mean(axis=0).round(3))
print(standardized_sample.std(axis=0).round(3))
```

### Actual Output

```text
Standardized Sample:
[[ 0.    -0.852 -0.551]
 [-1.225 -0.552 -0.852]
 [ 1.225  1.404  1.404]]

Sample column means: [0. 0. 0.]
Sample column stds: [1. 1. 1.]
```

### Explanation

- `StandardScaler()` creates a standardization scaler.
- `fit_transform()` calculates the column statistics and transforms the data.
- `mean(axis=0)` calculates the mean of each column.
- `std(axis=0)` calculates the standard deviation of each column.

## 7. Creating Features and Target

In Machine Learning, a dataset generally contains input features and target labels.

- `X`: Input features used by the model.
- `y`: Target labels that the model learns to predict.

### Python Example

```python
rng = np.random.default_rng(42)

X = rng.random((100, 5))
y = rng.integers(0, 2, 100)
```

### Explanation

- `default_rng(42)` creates a reproducible random number generator.
- `rng.random((100, 5))` creates 100 records with 5 features.
- `rng.integers(0, 2, 100)` creates 100 binary labels: 0 or 1.

## 8. Training and Testing Data Split

A dataset is commonly divided into training and testing portions.

### Training Data

Training data is used by the model to learn patterns.

### Testing Data

Testing data is used to evaluate how the trained model performs on data it has not trained on.

### 70/30 Split

In this lab:

- 70% of the data is used for training.
- 30% of the data is used for testing.

### Python Example

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)
```

### Explanation of Parameters

| Parameter | Purpose |
|---|---|
| `test_size=0.3` | Uses 30% for testing |
| `random_state=42` | Makes the split reproducible |
| `stratify=y` | Maintains similar class proportions in both sets |

### Actual Output

```text
Training Features Shape: (70, 5)
Testing Features Shape: (30, 5)
```

## 9. Data Leakage

Data leakage occurs when information from the testing dataset influences the training process.

For example, fitting a scaler on the entire dataset before splitting can allow information from the test set to influence preprocessing.

This can produce misleading evaluation results.

### Correct Approach

1. Split the dataset first.
2. Fit the scaler using training data only.
3. Transform training data.
4. Transform testing data using the same fitted scaler.

## 10. MinMaxScaler on Training and Testing Data

### Python Example

```python
minmax = MinMaxScaler().fit(X_train)

X_train_norm = minmax.transform(X_train)
X_test_norm = minmax.transform(X_test)
```

### Explanation

- `fit(X_train)` learns scaling information from training data.
- `transform(X_train)` normalizes the training data.
- `transform(X_test)` applies the same scaling to testing data.

### Actual Output

```text
Normalized train min/max: 0.0 1.0
Test normalized min/max: -0.002 1.055
```

The test values can fall slightly outside the 0 to 1 range because the test dataset may contain values outside the training data's minimum and maximum.

This is expected and does not indicate an error.

## 11. StandardScaler on Training and Testing Data

### Python Example

```python
std = StandardScaler().fit(X_train)

X_train_std = std.transform(X_train)
X_test_std = std.transform(X_test)
```

### Actual Output

```text
Standardized train mean: 0.0
Standardized train std: 1.0
```

The training features have a mean of approximately 0 and a standard deviation of approximately 1.

## 12. Project Directory Structure

```text
Lab-07-Data-Preprocessing-for-Deep-Learning/
│
├── python-practicals/
│   └── preprocess.py
│
├── evidence/
│   └── README.md
│
├── .gitignore
├── requirements.txt
└── README.md
```

## 13. Practical Execution

Navigate to the practical directory:

```bash
cd python-practicals
```

Run the Python script:

```bash
python3 preprocess.py
```

Verify the seven practical sections:

```bash
grep -n "^# ---- Step" preprocess.py
```

## 14. Key Differences: Normalization vs Standardization

| Normalization | Standardization |
|---|---|
| Uses MinMaxScaler | Uses StandardScaler |
| Scales features to a specified range | Centers features around mean 0 |
| Default range is 0 to 1 | Standard deviation is approximately 1 |
| Sensitive to minimum and maximum values | Sensitive to outliers as well |

## 15. Important Interview Questions

**Q1. What is data preprocessing?**

Data preprocessing is the process of preparing raw data before using it to train a Machine Learning or Deep Learning model.

**Q2. What is normalization?**

Normalization scales numerical features into a specified range, such as 0 to 1.

**Q3. What is MinMaxScaler?**

MinMaxScaler is a Scikit-learn preprocessing tool that scales each feature using its minimum and maximum values.

**Q4. What is standardization?**

Standardization transforms features to have a mean close to 0 and a standard deviation close to 1.

**Q5. What is StandardScaler?**

StandardScaler is a Scikit-learn tool used to standardize numerical features.

**Q6. What is train_test_split?**

It is a Scikit-learn function used to divide a dataset into training and testing sets.

**Q7. Why do we use random_state?**

It makes the random dataset split reproducible.

**Q8. What is stratify=y?**

It helps maintain similar target-class proportions in the training and testing datasets.

**Q9. What is data leakage?**

Data leakage occurs when information from outside the training data influences model training.

**Q10. Why do we fit the scaler only on training data?**

To prevent information from the test dataset from influencing the preprocessing process.

## 16. Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError` | Activate the virtual environment and install required packages |
| `externally-managed-environment` | Use a Python virtual environment |
| Wrong package version | Install the pinned package version |
| Incorrect file path | Check the current directory using `pwd` |
| Python syntax error | Review indentation, brackets, and spelling |
| Test scaling outside 0–1 | Expected when test values exceed training minimum or maximum |

## 17. Lab Conclusion

In this lab, I successfully completed the practical implementation of data preprocessing for Deep Learning.

I learned how to use NumPy and Scikit-learn to normalize and standardize numerical data, create features and target labels, and divide a dataset into training and testing portions.

I also learned why the scaler must be fitted only on training data to prevent data leakage.

All seven steps of the Python practical executed successfully without errors.

**Lab 07 – Data Preprocessing for Deep Learning: Completed.**
