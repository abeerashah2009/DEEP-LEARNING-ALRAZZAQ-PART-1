# Lab 22: Batch Normalization in Deep Networks

---

## 1. Objective

- Understand Batch Normalization.
- Learn how to add Batch Normalization layers to a neural network.
- Train models with and without Batch Normalization.
- Compare training convergence and model accuracy.

---

## 2. Environment Setup

- Operating System: Ubuntu 24.04
- Python: 3.12.3
- NumPy: 1.26.4
- Pandas: 2.2.2
- TensorFlow CPU: 2.17.0
- Keras: 3.15.1

> Training was performed using CPU.

---

## 3. Dataset

**Dataset:** MNIST Handwritten Digits

- Training images: 60,000
- Testing images: 10,000
- Image size: 28 × 28 pixels
- Number of classes: 10 (digits 0–9)
- Pixel values normalized between 0 and 1.

---

## 4. Model Architecture

### Model 1: With Batch Normalization

1. Input Layer (28 × 28)
2. Flatten Layer
3. Dense Layer (128 neurons, ReLU)
4. Batch Normalization
5. Dense Layer (64 neurons, ReLU)
6. Batch Normalization
7. Output Layer (10 neurons, Softmax)

**Total Parameters:** 110,154

### Model 2: Without Batch Normalization

- Same neural network architecture.
- Both Batch Normalization layers were removed.
- Used for comparison with Model 1.

---

## 5. Training Configuration

- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Evaluation Metric: Accuracy
- Epochs: 3
- Batch Size: 128
- Random Seed: 42

---

## 6. Validation Accuracy Comparison

| Epoch | With Batch Normalization | Without Batch Normalization |
|---|---:|---:|
| 1 | 96.18% | 95.10% |
| 2 | 96.85% | 96.37% |
| 3 | 96.80% | 96.96% |

---

## 7. Final Test Accuracy

| Model | Test Accuracy |
|---|---:|
| With Batch Normalization | 96.80% |
| Without Batch Normalization | 96.96% |

---

## 8. Observations

- Batch Normalization achieved higher validation accuracy during the first two epochs.
- The model without Batch Normalization achieved slightly higher accuracy in Epoch 3.
- Both models achieved approximately 97% test accuracy.
- The final accuracy difference was only 0.16 percentage points.

> Batch Normalization improved early validation performance in this experiment, but did not produce higher final test accuracy.

---

## 9. Conclusion

- Batch Normalization helps stabilize neural network training.
- It can improve early convergence.
- Its impact on final accuracy depends on the model and training conditions.
- It does not guarantee higher accuracy in every experiment.

---

## 10. Project Files

- `bn_lab.py` — Python implementation of both models.
- `README.md` — Lab documentation.
- `.gitignore` — Excludes virtual environment and Python cache files.

---

## 11. Run the Lab

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Run the Python program:

```bash
python bn_lab.py
```

> MNIST is downloaded automatically if it is not already cached.

---

**Lab Status:** Implementation, training, evaluation, and model comparison completed.
