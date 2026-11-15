# Lab 33: Transfer Learning for Image Classification

## 1. Lab Overview

In this lab, I implemented Transfer Learning using a pre-trained VGG16 Convolutional Neural Network (CNN).

I used ImageNet pre-trained weights and modified the original architecture by adding custom classification layers to recognize five different flower categories.

The model was trained and evaluated using the TensorFlow Flower Photos dataset.

## 2. Lab Objectives

- Understand the concept of Transfer Learning.
- Load a pre-trained VGG16 model.
- Use ImageNet pre-trained weights.
- Remove the original classification layers.
- Freeze the pre-trained layers.
- Add custom classification layers.
- Train the model on a new image dataset.
- Evaluate model performance using accuracy and loss.

## 3. Environment and Tools

| Component | Version / Details |
|---|---|
| Operating System | Ubuntu 24.04 LTS |
| Python | 3.12.3 |
| TensorFlow CPU | 2.17.0 |
| NumPy | 1.26.4 |
| Keras | Included with TensorFlow |
| Model | VGG16 |
| Pre-trained weights | ImageNet |
| Dataset | TensorFlow Flower Photos |
| Hardware | CPU |

## 4. Project Structure

```text
Lab-33-Transfer-Learning-for-Image-Classification/
│
├── load_model.py
├── build_model.py
├── train.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/                 # Downloaded dataset (not committed)
├── venv/                 # Python virtual environment (not committed)
├── flower_photos.tgz     # Dataset archive (not committed)
│
└── *.log                 # Execution logs (not committed)
```

## 5. Environment Setup

Created and activated a Python virtual environment.

```bash
python3 -m venv venv
source venv/bin/activate
```

Installed the required libraries.

```bash
pip install --upgrade pip
pip install "tensorflow-cpu==2.17.0" "numpy<2.0"
```

Verified the installed versions.

```bash
python -c "import tensorflow as tf, numpy as np; print(tf.__version__, np.__version__)"
```

**Verified output:**

```text
TensorFlow: 2.17.0
NumPy: 1.26.4
GPU devices: []
```

## 6. Task 1: Load the Pre-trained VGG16 Model

### Description

VGG16 is a pre-trained Convolutional Neural Network trained on the ImageNet dataset.

Transfer Learning allows us to reuse features learned by an existing model instead of training a complete neural network from scratch.

### Implementation

Created `load_model.py`.

```python
import tensorflow as tf
from tensorflow.keras.applications import VGG16

base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.summary()
```

### Important Parameters

- `weights='imagenet'`: Loads pre-trained ImageNet weights.
- `include_top=False`: Removes the original classification layers.
- `input_shape=(224, 224, 3)`: Accepts RGB images of size 224 × 224.

### Verification

Executed:

```bash
python load_model.py
```

**Verified model parameters:**

| Parameter | Result |
|---|---:|
| Total parameters | 14,714,688 |
| Trainable parameters | 14,714,688 |
| Non-trainable parameters | 0 |

**Task 1 completed successfully.**

## 7. Task 2: Modify the Pre-trained Architecture

### Description

The original VGG16 classification layers were removed.

I froze the pre-trained VGG16 layers and added custom layers to classify five flower categories.

### Dataset

Downloaded the TensorFlow Flower Photos dataset.

```bash
wget https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
mkdir -p data
tar -xzf flower_photos.tgz -C data
```

### Dataset Categories

| Flower Category | Number of Images |
|---|---:|
| Daisy | 633 |
| Dandelion | 898 |
| Roses | 641 |
| Sunflowers | 699 |
| Tulips | 799 |
| **Total** | **3,670** |

### Implementation

Created `build_model.py`.

```python
import keras
from keras import layers
from keras.applications import VGG16

num_classes = 5

base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)

output_layer = layers.Dense(
    num_classes,
    activation='softmax'
)(x)

model = keras.Model(
    inputs=base_model.input,
    outputs=output_layer
)

model.summary()
```

### Explanation of Custom Layers

**1. GlobalAveragePooling2D**

Reduces the spatial dimensions of feature maps and produces a compact feature vector.

**2. Dense Layer**

Contains 256 neurons with ReLU activation to learn useful classification patterns.

**3. Dropout Layer**

Uses a dropout rate of 0.5 to help reduce overfitting during training.

**4. Output Layer**

Contains five neurons with softmax activation, one for each flower category.

### Model Verification

Executed:

```bash
python build_model.py
```

**Verified model parameters:**

| Parameter | Result |
|---|---:|
| Total parameters | 14,847,301 |
| Trainable parameters | 132,613 |
| Non-trainable parameters | 14,714,688 |

The pre-trained VGG16 layers were successfully frozen.

**Task 2 completed successfully.**

## 8. Task 3: Compile, Train, and Evaluate the Model

### Dataset Preparation

The dataset was divided into:

- Training data: 80% (2,936 images)
- Validation data: 20% (734 images)

Image preprocessing was performed using the VGG16 `preprocess_input` function.

To keep CPU training manageable, the training script used 20 batches per epoch and 10 validation batches.

### Model Compilation

The model was compiled using:

- Optimizer: Adam
- Loss function: Categorical Crossentropy
- Evaluation metric: Accuracy

### Training Configuration

| Setting | Value |
|---|---|
| Epochs | 3 |
| Batch size | 32 |
| Training steps per epoch | 20 |
| Validation batches | 10 |
| Number of classes | 5 |

### Training Results

Actual output from the completed training run:

| Metric | Epoch 1 | Epoch 2 | Epoch 3 |
|---|---:|---:|---:|
| Training Accuracy | 43.91% | 72.81% | 82.19% |
| Validation Accuracy | 71.25% | 83.44% | 81.25% |
| Training Loss | 3.8573 | 1.2720 | 0.5764 |
| Validation Loss | 1.0707 | 0.7206 | 0.7516 |

### Final Evaluation

```text
Test Accuracy: 81.25%
```

**Final test accuracy: 81.25%**

The model successfully learned to classify flower images using pre-trained VGG16 features and custom classification layers.

**Task 3 completed successfully.**

## 9. Important Concepts Learned

### Transfer Learning

Transfer Learning reuses knowledge from a pre-trained model for a new task.

### Pre-trained Model

A neural network that has already been trained on a large dataset.

### VGG16

A 16-layer deep convolutional neural network commonly used for image classification and feature extraction.

### Feature Extraction

Using the learned convolutional layers of a pre-trained model to extract useful image features.

### Frozen Layers

Layers whose weights are not updated during training.

### Softmax Activation

Converts output scores into class probabilities.

### Overfitting

When a model performs well on training data but does not generalize well to unseen data.

## 10. Important Commands

```bash
# Activate virtual environment
source venv/bin/activate

# Check TensorFlow version
python -c "import tensorflow as tf; print(tf.__version__)"

# Check Python syntax
python -m py_compile load_model.py
python -m py_compile build_model.py
python -m py_compile train.py

# Load VGG16
python load_model.py

# Build custom model
python build_model.py

# Train and evaluate model
python train.py
```

## 11. Conclusion

In this lab, I successfully implemented Transfer Learning using the VGG16 pre-trained model.

I loaded ImageNet weights, removed the original classification layers, froze the pre-trained layers, and added custom classification layers for five flower categories.

I trained and evaluated the model using TensorFlow and Keras on the Flower Photos dataset.

The final evaluation achieved **81.25% test accuracy**.

This practical exercise improved my understanding of Transfer Learning, CNN architecture, feature extraction, model training, and image classification.

**Lab 33: Completed Successfully.**
