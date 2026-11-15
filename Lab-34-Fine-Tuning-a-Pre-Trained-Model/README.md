# Lab 34: Fine-Tuning a Pre-Trained Model

## 1. Lab Overview

In this lab, I learned how to use **Transfer Learning and Fine-Tuning** with a pre-trained VGG16 model.

I used the VGG16 model with ImageNet weights and adapted it for image classification using the CIFAR-10 dataset.

The lab was completed in two training stages:

1. Baseline training with the VGG16 base frozen.
2. Fine-tuning after unfreezing only VGG16 block5.

The purpose was to compare model accuracy before and after fine-tuning.

---

## 2. Lab Objectives

- Understand Transfer Learning.
- Understand Fine-Tuning.
- Load a pre-trained VGG16 model.
- Use ImageNet pre-trained weights.
- Freeze the convolutional base.
- Add a new classification head.
- Load and preprocess the CIFAR-10 dataset.
- Train the classification head.
- Unfreeze only block5.
- Fine-tune the model with a lower learning rate.
- Compare baseline and final test accuracy.

---

## 3. Environment and Tools

| Component | Version / Configuration |
|---|---|
| Operating System | Ubuntu 24.04 |
| Python | 3.12.3 |
| TensorFlow CPU | 2.17.0 |
| Keras | 3.15.1 |
| NumPy | 1.26.4 |
| Model | VGG16 |
| Dataset | CIFAR-10 |
| Training Device | CPU |

### Python Virtual Environment

A Python virtual environment was created to keep project dependencies separate.

Activate the environment:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 4. Important Concepts and Notes

### A. What is Transfer Learning?

Transfer Learning is a machine learning technique in which we use knowledge learned by an already-trained model for a new task.

Instead of training a complete neural network from the beginning, we reuse an existing model.

**Example:**

A VGG16 model trained on ImageNet already knows how to recognize image features such as edges, shapes, and textures.

We can reuse this knowledge to classify images from CIFAR-10.

### B. What is Fine-Tuning?

Fine-Tuning means taking a pre-trained model and training some of its existing layers on a new dataset.

It allows the model to adjust previously learned features for a new classification task.

### C. What is VGG16?

VGG16 is a pre-trained Convolutional Neural Network (CNN).

It contains 19 layers in the convolutional base used in this lab:

- Input layer
- Block1
- Block2
- Block3
- Block4
- Block5

It uses ImageNet weights to provide previously learned image features.

### D. What is ImageNet?

ImageNet is a large image dataset used to train image recognition models.

In this lab, VGG16 uses pre-trained ImageNet weights.

### E. What is CIFAR-10?

CIFAR-10 is an image classification dataset containing 10 classes.

Each image has dimensions:

```text
32 x 32 x 3
```

The three channels represent Red, Green, and Blue (RGB).

For this lab, I used:

- 10,000 training images
- 2,000 testing images
- 10 image classes

### F. What is Freezing a Layer?

Freezing a layer means preventing its weights from being updated during training.

In the baseline stage, the complete VGG16 base was frozen.

Only the new classification head was trained.

### G. What is Unfreezing?

Unfreezing means allowing selected layers to update their weights during training.

In this lab, only block5 was unfrozen.

Blocks 1, 2, 3, and 4 remained frozen.

### H. What is a Learning Rate?

The learning rate controls how much model weights change during training.

Two learning rates were used:

- Baseline training: `0.001` (1e-3)
- Fine-tuning: `0.00001` (1e-5)

A lower learning rate was used during fine-tuning to make smaller weight updates.

---

## 5. Project Structure

```text
Lab-34-Fine-Tuning-a-Pre-Trained-Model/
│
├── ft.py
├── unfreeze.part
├── requirements.txt
├── README.md
├── .gitignore
├── run.log
└── venv/
```

Notes:

- `ft.py`: Main Python training script.
- `unfreeze.part`: Contains the code used to unfreeze block5.
- `requirements.txt`: Project dependencies.
- `README.md`: Lab documentation.
- `.gitignore`: Excludes unnecessary files from Git.
- `run.log`: Training output; excluded from Git.
- `venv/`: Python virtual environment; excluded from Git.

---

## 6. VGG16 Model Configuration

The VGG16 model was loaded using:

```python
base = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(32, 32, 3)
)
```

Explanation:

- `weights='imagenet'`: Loads pre-trained ImageNet weights.
- `include_top=False`: Removes the original classification head.
- `input_shape=(32, 32, 3)`: Accepts CIFAR-10 image dimensions.

The complete pre-trained base was initially frozen:

```python
base.trainable = False
```

### New Classification Head

A new classification head was added:

```python
model = keras.Sequential([
    base,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax'),
])
```

Explanation:

- Flatten: Converts feature maps into a one-dimensional vector.
- Dense: Learns classification features.
- Dropout: Helps reduce overfitting.
- Softmax: Produces probabilities for 10 classes.

---

## 7. CIFAR-10 Dataset Preparation

The dataset was loaded using:

```python
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
```

A smaller subset was selected:

```python
x_train, y_train = x_train[:10000], y_train[:10000]
x_test, y_test = x_test[:2000], y_test[:2000]
```

Images were converted to float32 and preprocessed for VGG16:

```python
x_train = preprocess_input(x_train.astype('float32'))
x_test = preprocess_input(x_test.astype('float32'))
```

Verified dataset shapes:

```text
Train: (10000, 32, 32, 3)
Test:  (2000, 32, 32, 3)
```

---

## 8. Stage 1: Baseline Training

In the first stage, the VGG16 base remained frozen.

The model was compiled using:

```python
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

Training configuration:

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | Sparse Categorical Crossentropy |
| Epochs | 3 |
| Batch Size | 64 |
| Validation Data | 2,000 images |

Training code:

```python
model.fit(
    x_train,
    y_train,
    epochs=3,
    batch_size=64,
    validation_data=(x_test, y_test),
    verbose=2
)
```

### Baseline Training Results

| Epoch | Training Accuracy | Validation Accuracy |
|---|---:|---:|
| 1 | 37.49% | 50.40% |
| 2 | 45.80% | 53.25% |
| 3 | 52.04% | 55.45% |

Baseline test accuracy:

**55.45%**

Trainable weight tensors:

**4**

Trainable parameters:

**133,898**

---

## 9. Stage 2: Fine-Tuning VGG16

After baseline training, I unfroze only block5.

The following code was used:

```python
base.trainable = True

for layer in base.layers:
    layer.trainable = layer.name.startswith('block5')
```

This kept blocks 1 to 4 frozen and allowed block5 to train.

### Recompile the Model

The model was recompiled using a lower learning rate:

```python
model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

Fine-tuning configuration:

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.00001 |
| Epochs | 3 |
| Batch Size | 64 |
| Unfrozen Block | block5 |

### Fine-Tuning Results

| Epoch | Training Accuracy | Validation Accuracy |
|---|---:|---:|
| 1 | 57.79% | 58.10% |
| 2 | 63.20% | 58.85% |
| 3 | 66.01% | 59.45% |

Fine-tuning trainable weight tensors:

**10**

Trainable parameters after unfreezing:

**7,213,322**

---

## 10. Baseline vs Fine-Tuning Comparison

| Metric | Baseline | Fine-Tuning |
|---|---:|---:|
| VGG16 Base | Frozen | Block5 unfrozen |
| Trainable Weight Tensors | 4 | 10 |
| Trainable Parameters | 133,898 | 7,213,322 |
| Training Epochs | 3 | 3 |
| Learning Rate | 0.001 | 0.00001 |
| Test Accuracy | 55.45% | 59.45% |

### Accuracy Improvement

```text
Initial Test Accuracy = 0.5545

Final Test Accuracy = 0.5945

Change = +0.0400
```

The model achieved an improvement of **4.00 percentage points** after fine-tuning.

---

## 11. Final Model Summary

| Parameter | Verified Result |
|---|---:|
| Total Parameters | 14,848,586 |
| Trainable Parameters | 7,213,322 |
| Non-Trainable Parameters | 7,635,264 |
| Frozen Base Trainable Tensors | 4 |
| Fine-Tuned Trainable Tensors | 10 |
| Final Test Accuracy | 59.45% |

---

## 12. Important Commands Used

### Activate Virtual Environment

```bash
source venv/bin/activate
```

### Check Python Version

```bash
python --version
```

### Check TensorFlow and Keras Versions

```bash
python -c "import tensorflow as tf; print(tf.__version__); print(tf.keras.__version__)"
```

### Check Python Syntax

```bash
python -m py_compile ft.py
```

### Run the Complete Training Script

```bash
python ft.py
```

### Check Training Results

```bash
grep -E 'accuracy|trainable tensors|Initial test accuracy|Final test accuracy|Change:' run.log
```

---

## 13. What I Learned

After completing this lab, I learned:

1. How to load a pre-trained VGG16 model.
2. How to reuse ImageNet weights.
3. How to freeze a pre-trained model.
4. How to add a new classification head.
5. How to load and preprocess CIFAR-10 images.
6. How to train a classification head.
7. How to unfreeze selected layers.
8. How to fine-tune a model using a lower learning rate.
9. How to compare baseline and fine-tuned accuracy.
10. How fine-tuning can improve image classification performance.

---

## 14. Conclusion

In this lab, I implemented Transfer Learning and Fine-Tuning using VGG16 and CIFAR-10.

First, I trained a new classification head while keeping the complete VGG16 base frozen. The baseline test accuracy was 55.45%.

After that, I unfroze only block5 and trained the model again using a lower learning rate. The final test accuracy improved to 59.45%.

The experiment demonstrated how fine-tuning selected layers of a pre-trained model can improve its performance on a new image classification task.

**Lab 34 completed successfully.**
