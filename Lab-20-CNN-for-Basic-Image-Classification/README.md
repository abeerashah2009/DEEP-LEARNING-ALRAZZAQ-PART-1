# Lab 20: CNN for Basic Image Classification

## 1. Objective

The objective of this lab is to understand and implement a Convolutional Neural Network (CNN) using TensorFlow and Keras.

In this lab, I built and trained a CNN model using the CIFAR-10 image dataset and evaluated its classification performance.

## 2. Environment

- Operating System: Ubuntu 24.04.3 LTS
- Python: 3.12.3
- TensorFlow CPU: 2.16.2
- NumPy: 1.26.4
- Matplotlib: 3.11.2
- Virtual Environment: Python venv

**Environment Note:** The provided lab prerequisites mention CentOS Stream 9. The available lab VM was running Ubuntu 24.04.3 LTS, so package and environment setup was adapted accordingly.

## 3. Project Files

| File | Description |
|---|---|
| `cnn.py` | CNN architecture, dataset loading, preprocessing, training and evaluation |
| `requirements.txt` | Required Python packages |
| `cnn_cifar10.keras` | Saved trained CNN model |

## 4. CNN Architecture

The model uses the following layers:

1. Input layer: 32 × 32 RGB images
2. Conv2D: 32 filters, 3 × 3 kernel, ReLU activation
3. MaxPooling2D: 2 × 2 pool size
4. Conv2D: 64 filters, 3 × 3 kernel, ReLU activation
5. MaxPooling2D: 2 × 2 pool size
6. Flatten layer
7. Dense layer: 64 neurons, ReLU activation
8. Output layer: 10 neurons, Softmax activation

**Total trainable parameters:** 167,562

## 5. Dataset

The model was trained and tested using the CIFAR-10 dataset.

- Training images: 50,000
- Testing images: 10,000
- Image dimensions: 32 × 32 pixels
- Color channels: 3 (RGB)
- Classes: 10

## 6. Data Preprocessing

The following preprocessing steps were performed:

- Normalized image pixel values by dividing by 255.0.
- Converted class labels into one-hot encoded format using `to_categorical()`.

The normalized pixel range was verified as 0.0 to 1.0.

## 7. Model Compilation

The model was compiled using:

- Optimizer: Adam
- Loss function: Categorical Crossentropy
- Evaluation metric: Accuracy

## 8. Model Training

Training configuration:

- Epochs: 10
- Batch size: 64
- Validation split: 10%

The model completed all 10 training epochs successfully.

Second training run results:

- Final training accuracy: 75.52%
- Final validation accuracy: 70.50%

## 9. Model Evaluation

The model was evaluated using the separate CIFAR-10 test dataset.

| Metric | Result |
|---|---:|
| Test Accuracy | 69.68% |
| Test Loss | 0.9159 |

## 10. Saved Model

The trained model was saved in Keras format:

`cnn_cifar10.keras`

File size: approximately 2.0 MB.

## 11. How to Run

Activate the virtual environment:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the CNN script:

```bash
python cnn.py
```

The script loads CIFAR-10, preprocesses the data, trains the CNN, evaluates test accuracy and saves the model.

**Note:** Running the complete script again starts a new training run.

## 12. Key Learning Outcomes

Through this lab, I learned how to:

- Build a CNN using TensorFlow and Keras.
- Understand convolution and max-pooling layers.
- Load the CIFAR-10 dataset.
- Normalize image data and encode labels.
- Compile and train a deep learning model.
- Evaluate classification accuracy.
- Save a trained model in `.keras` format.

## 13. Conclusion

I successfully implemented and trained a Convolutional Neural Network for basic image classification using the CIFAR-10 dataset.

The model achieved **69.68% test accuracy** and was saved successfully as `cnn_cifar10.keras`.
