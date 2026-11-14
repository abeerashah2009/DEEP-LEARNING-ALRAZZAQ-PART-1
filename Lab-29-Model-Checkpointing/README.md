# Lab 29: Model Checkpointing

---

## 1. Introduction

Model Checkpointing is a technique used in Deep Learning to save a model's weights during training.

It allows us to preserve the best-performing model instead of losing its progress when training finishes or stops.

In Keras, we use the `ModelCheckpoint` callback to save the best model automatically.

## 2. Objectives

- Understand Model Checkpointing.
- Learn how to use the Keras `ModelCheckpoint` callback.
- Save the best model weights during training.
- Reload previously saved model weights.
- Evaluate model performance after loading the checkpoint.

## 3. Technologies Used

- Python 3.12
- TensorFlow CPU 2.17.0
- Keras 3.15.1
- Scikit-learn 1.9.1
- NumPy 1.26.4
- Ubuntu Linux

## 4. Dataset Information

This lab uses the Scikit-learn Handwritten Digits Dataset.

Dataset details:

- Total samples: 1,797
- Image dimensions: 8 × 8 pixels
- Features per image: 64
- Number of classes: 10
- Classes: Digits 0 to 9
- Training samples: 1,437
- Testing samples: 360

Pixel values are normalized from 0–16 to 0–1.

### Dataset Split

- 80% Training Data
- 20% Testing Data
- Random State: 42
- Stratification: Enabled

## 5. Model Architecture

The neural network contains three Dense layers.

| Layer | Neurons | Activation |
|---|---:|---|
| Input | 64 | None |
| Hidden Layer 1 | 64 | ReLU |
| Hidden Layer 2 | 64 | ReLU |
| Output Layer | 10 | Softmax |

Total Parameters: 8,970

### Optimizer

Adam

### Loss Function

Sparse Categorical Crossentropy

### Evaluation Metric

Accuracy

## 6. Project Structure

```text
Lab-29-Model-Checkpointing/
│
├── data.py
├── model_def.py
├── train.py
├── reload.py
├── best.weights.h5
├── requirements.txt
└── README.md
```

## 7. File Descriptions

**data.py**

Loads the handwritten digits dataset, normalizes pixel values, and splits the data into training and testing sets.

**model_def.py**

Defines the neural network architecture and compiles the model.

**train.py**

Trains the model and uses the ModelCheckpoint callback to save the best weights.

**reload.py**

Loads the saved weights into a new model and evaluates its performance.

**best.weights.h5**

Contains the best saved model weights.

**requirements.txt**

Contains the Python dependencies required for this project.

## 8. Understanding ModelCheckpoint

Keras provides the `ModelCheckpoint` callback to save model information during training.

Important parameters:

| Parameter | Purpose |
|---|---|
| filepath | Location and name of the checkpoint file |
| monitor | Metric used to monitor model performance |
| save_best_only | Saves only when performance improves |
| save_weights_only | Saves only model weights |
| mode | Defines whether to maximize or minimize the monitored metric |
| verbose | Displays checkpoint messages |

### Configuration Used in This Lab

```python
checkpoint = ModelCheckpoint(
    "best.weights.h5",
    monitor="val_accuracy",
    save_best_only=True,
    save_weights_only=True,
    mode="max",
    verbose=1
)
```

Explanation:

- `best.weights.h5`: Name of the saved weights file.
- `monitor="val_accuracy"`: Monitors validation accuracy.
- `save_best_only=True`: Saves weights only when validation accuracy improves.
- `save_weights_only=True`: Saves model weights instead of the complete model.
- `mode="max"`: Higher validation accuracy is better.
- `verbose=1`: Displays checkpoint messages.

## 9. Training Process

The model was trained for 10 epochs.

During training, the ModelCheckpoint callback monitored validation accuracy.

Whenever validation accuracy improved, the best model weights were saved in:

`best.weights.h5`

### Training Results

| Metric | Result |
|---|---:|
| Total Epochs | 10 |
| Best Validation Accuracy | 94.44% |
| Final Training Accuracy | 96.95% |
| Final Validation Loss | 0.1953 |

The best validation accuracy was achieved during Epoch 9.

During Epoch 10, validation accuracy did not improve, so the previous best checkpoint was retained.

## 10. Checking the Checkpoint File

Command:

```bash
ls -lh best.weights.h5
```

Result:

```text
-rw-rw-r-- 1 ubuntu ubuntu 128K best.weights.h5
```

This confirms that the checkpoint file was successfully created.

## 11. Reloading Saved Weights

The saved weights were loaded into a newly created model.

Command:

```python
model.load_weights("best.weights.h5")
```

This restores the saved model weights.

The model was evaluated before and after loading the checkpoint.

### Reload Results

| Evaluation | Accuracy |
|---|---:|
| Before Loading Weights | 10.28% |
| After Loading Weights | 94.72% |

### Result Explanation

Before loading the saved weights, the model had random weights and achieved 10.28% accuracy.

After loading the best checkpoint, the model achieved 94.72% test accuracy.

This confirms that the saved weights were successfully restored.

## 12. Requirements

The following packages were used:

```text
tensorflow-cpu==2.17.0
scikit-learn==1.9.1
numpy==1.26.4
```

Install dependencies using:

```bash
pip install -r requirements.txt
```

## 13. How to Execute the Project

### Step 1: Activate Virtual Environment

```bash
source ~/lab-venv/bin/activate
```

### Step 2: Install Requirements

```bash
pip install -r requirements.txt
```

### Step 3: Train the Model

```bash
python train.py
```

### Step 4: Check Saved Weights

```bash
ls -lh best.weights.h5
```

### Step 5: Reload and Evaluate

```bash
python reload.py
```

## 14. Important Learning Points

1. Model Checkpointing saves model weights during training.
2. Keras provides the `ModelCheckpoint` callback.
3. Validation accuracy can be monitored during training.
4. `save_best_only=True` preserves the best-performing checkpoint.
5. Saved weights can be reloaded using `load_weights()`.
6. A model can be evaluated after restoring its saved weights.

## 15. Conclusion

In this lab, I implemented Model Checkpointing using TensorFlow and Keras.

I trained a neural network on the handwritten digits dataset and used the ModelCheckpoint callback to save the best model weights.

I successfully reloaded the saved weights and evaluated the model.

The model achieved:

- Best validation accuracy: 94.44%
- Test accuracy after loading checkpoint: 94.72%

This practical exercise helped me understand how to save, restore, and evaluate the best-performing model during Deep Learning training.

---

**Lab 29: Model Checkpointing — Completed**
