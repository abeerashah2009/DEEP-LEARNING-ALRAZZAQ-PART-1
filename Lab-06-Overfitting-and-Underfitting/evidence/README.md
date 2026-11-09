# Lab 06 – Evidence: Overfitting and Underfitting

## 1. Lab Overview

This evidence folder contains the results and observations from Lab 06. The practical demonstrates overfitting, underfitting, training and validation errors, dropout, and Ridge regularization.

**Environment:**
- Operating System: Ubuntu 24.04 LTS
- Python: 3.12
- Scikit-learn: 1.5.2
- NumPy: 1.26.4
- Matplotlib: 3.9.2

## 2. Dataset Preparation

**Dataset:** Scikit-learn Diabetes Dataset

- Selected feature: BMI
- Total selected samples: 50
- Training samples: 25
- Validation samples: 25
- Random state: 42

**Verified output:**

```text
Training data: (25, 1)
Validation data: (25, 1)
```

## 3. Simple Model Comparison

Two polynomial regression models were compared.

| Model | Training MSE | Validation MSE |
|---|---:|---:|
| Degree 1 | 3994.7 | 4602.9 |
| Degree 10 | 1791.6 | 4139436675908272.0 |

**Observation:**

- Degree 1 has relatively high training and validation errors.
- Degree 10 has lower training error but extremely high validation error.
- This demonstrates severe overfitting in the degree-10 model.

## 4. Polynomial Degree Sweep

Polynomial models from degree 1 to degree 10 were tested.

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

**Best degree:** 2

The degree-2 model achieved the lowest validation MSE of 4159.9 in this experiment.

**Verification:**

```text
Best degree (lowest val MSE): 2
Assertions passed
```

## 5. Training and Validation Loss Graph

Graph file: `../loss.png`

The graph displays training and validation Mean Squared Error (MSE) for polynomial degrees 1–10.

**Observation:**

- Training error generally decreases as model complexity increases.
- Validation error increases sharply at higher degrees.
- The widening gap indicates overfitting.

## 6. Dropout Demonstration

Dropout probability: 0.5

**Output:**

```text
[2. 0. 2. 2. 0. 2. 2. 2. 0. 0.]
```

**Observation:**

- Four values were set to zero.
- Six values remained active.
- Active values were scaled to 2.

This is a mathematical illustration of dropout, not a trained neural network.

## 7. Ridge Regularization Experiment

A degree-10 polynomial model was evaluated with and without Ridge regularization.

| Model | Validation MSE |
|---|---:|
| Plain Linear Regression | 4139436675908272.0 |
| Ridge (alpha = 10.0) | 20969454.8 |

**Observation:**

Ridge regularization significantly reduced the validation error by penalizing large model coefficients and controlling model complexity.

However, the Ridge validation error was still higher than the best degree-2 model.

## 8. Practical Files

| File | Purpose |
|---|---|
| `data.py` | Loads and prepares the dataset |
| `simple.py` | Compares degree-1 and degree-10 models |
| `sweep.py` | Tests polynomial degrees 1–10 |
| `ridge.py` | Compares plain regression with Ridge |
| `table.txt` | Stores polynomial sweep results |
| `loss.png` | Training and validation loss graph |

## 9. Final Conclusion

This lab demonstrated how model complexity affects training and validation performance.

The experiments showed that a complex model can achieve low training error while performing poorly on unseen data. This is overfitting.

Dropout and Ridge regularization were explored as techniques to control overfitting. The degree-2 polynomial model achieved the best validation result in this experiment.

**Lab 06 practical evidence completed.**
