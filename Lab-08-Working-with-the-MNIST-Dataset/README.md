
# Lab 08 - Working with the MNIST Dataset

## 1. Overview

This practical focuses on loading, understanding, preprocessing, and visualizing the MNIST handwritten digit dataset using Python, TensorFlow, and Matplotlib.

MNIST is a popular dataset used in machine learning and deep learning for handwritten digit recognition.

It contains grayscale images of handwritten digits from 0 to 9.

## 2. Learning Objectives

- Understand the structure of the MNIST dataset.
- Load training and testing data using TensorFlow Keras.
- Understand image dimensions and labels.
- Normalize image pixel values.
- Visualize handwritten digit images using Matplotlib.
- Save generated output as evidence.

## 3. Technologies Used

| Technology | Version | Purpose |
|---|---|---|
| Python | 3.12.3 | Programming language |
| TensorFlow | 2.17.1 | Dataset loading |
| Matplotlib | 3.9.4 | Data visualization |
| NumPy | 1.26.4 | Numerical operations |
| Ubuntu | 24.04 | Operating system |

## 4. Understanding the MNIST Dataset

MNIST stands for Modified National Institute of Standards and Technology.

The dataset contains:

| Dataset | Number of Images |
|---|---:|
| Training dataset | 60,000 |
| Testing dataset | 10,000 |
| Total | 70,000 |

Each image:
- Has dimensions of 28 x 28 pixels.
- Uses grayscale pixel values.
- Represents one handwritten digit from 0 to 9.

### Dataset Shapes

| Data | Shape |
|---|---|
| Training images | (60000, 28, 28) |
| Training labels | (60000,) |
| Testing images | (10000, 28, 28) |
| Testing labels | (10000,) |

The first number represents the number of samples. The remaining two numbers represent image height and width.

Labels contain the correct digit for each image.

## 5. Environment Setup

A separate Python virtual environment was created outside the Git repository.

Create the environment:

```bash
python3 -m venv ~/mnist-lab-venv
```

Activate it:

```bash
source ~/mnist-lab-venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 6. Loading the MNIST Dataset

The dataset was loaded using TensorFlow Keras:

```python
from tensorflow.keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
```

The function returns four arrays:

1. Training images
2. Training labels
3. Testing images
4. Testing labels

The dataset is downloaded automatically if it is not already available in the local cache.

## 7. Data Normalization

Before normalization, image pixel values range from 0 to 255.

Normalization converts these values into a range between 0 and 1.

### Formula

Normalized Pixel Value = Original Pixel Value / 255

### Python Implementation

```python
train_images = train_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0
```

The `astype("float32")` operation converts pixel values into floating-point numbers.

### Verified Results

```text
Max pixel value after normalization: 1.0
Min pixel value after normalization: 0.0
```

## 8. Visualizing Handwritten Digits

Matplotlib was used to visualize the first 10 training images.

The visualization:
- Displays 10 handwritten digit images.
- Uses grayscale colors.
- Displays the corresponding label above each image.
- Saves the output as a PNG image.

Generated file:

`evidence/samples.png`

Image dimensions: 1000 x 500 pixels.

The Matplotlib `Agg` backend was selected because this practical was executed on a remote Ubuntu server without a graphical desktop.

## 9. Practical Execution

Run the practical from the Lab 08 directory:

```bash
python3 python-practicals/mnist_lab.py
```

### Actual Verified Output

```text
Train images shape: (60000, 28, 28)
Train labels shape: (60000,)
Test images shape: (10000, 28, 28)
Test labels shape: (10000,)

Max pixel value after normalization: 1.0
Min pixel value after normalization: 0.0

First 10 labels: [5, 0, 4, 1, 9, 2, 1, 3, 1, 4]
```

The visualization was successfully saved to:

`evidence/samples.png`

## 10. Project Structure

```text
Lab-08-Working-with-the-MNIST-Dataset/
│
├── python-practicals/
│   └── mnist_lab.py
│
├── evidence/
│   ├── README.md
│   └── samples.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

## 11. Important Concepts Learned

### Dataset
A dataset is a collection of information used to train and evaluate machine learning models.

### Training Data
Training data is used by a model to learn patterns.

### Testing Data
Testing data is used to evaluate how well a trained model performs on unseen examples.

### Labels
Labels are the correct answers associated with individual images.

### Normalization
Normalization changes numerical values into a smaller and consistent range.

### Visualization
Visualization helps us understand data through graphical representation.

## 12. Troubleshooting

### TensorFlow Cannot Find CUDA Drivers

The practical can run using the CPU. An NVIDIA GPU is not required for this exercise.

### MNIST Download Takes Time

The dataset is downloaded during the first execution. Later executions can use the cached dataset.

### Matplotlib Display Error

The `Agg` backend allows plots to be saved as image files without requiring a graphical display.

## 13. Interview Questions and Answers

**Q1. What is MNIST?**

MNIST is a dataset containing 70,000 grayscale handwritten digit images from 0 to 9.

**Q2. What are the dimensions of each MNIST image?**

Each image has dimensions of 28 x 28 pixels.

**Q3. How many training images are available in MNIST?**

There are 60,000 training images.

**Q4. How many testing images are available?**

There are 10,000 testing images.

**Q5. What is normalization?**

Normalization converts numerical values into a consistent range, such as 0 to 1.

**Q6. Why do we divide pixel values by 255?**

Because grayscale pixel values range from 0 to 255. Dividing by 255 converts them into the range 0 to 1.

**Q7. What is the purpose of training data?**

Training data helps a model learn patterns from examples.

**Q8. What is the purpose of testing data?**

Testing data evaluates model performance on data that was not used for training.

**Q9. What is Matplotlib?**

Matplotlib is a Python library used to create graphs and visualizations.

**Q10. Why was the Agg backend used?**

It allows Matplotlib to save images without requiring a graphical display.

## 14. Conclusion

In this practical, I successfully loaded the MNIST handwritten digit dataset using TensorFlow Keras.

I explored the training and testing dataset structures, checked image dimensions, normalized pixel values, and visualized the first 10 handwritten digits using Matplotlib.

I also generated and verified the `samples.png` output file.

This practical provided hands-on experience with dataset loading, preprocessing, normalization, and visualization, which are important steps in deep learning workflows.
