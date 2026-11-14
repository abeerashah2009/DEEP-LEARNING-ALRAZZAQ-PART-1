# Lab 21: Data Augmentation for Images

## Overview

This project demonstrates image augmentation using TensorFlow, Keras, and Albumentations.

A synthetic image dataset containing circles and squares was generated. Different augmentation techniques were applied to training images, and two CNN models were trained to compare performance with and without augmentation.

## Objectives

- Understand image augmentation.
- Apply random rotation, flipping, and translation.
- Generate augmented image batches.
- Visualize augmented images.
- Train CNN models with and without augmentation.
- Compare test accuracy.

## Environment

| Component | Version |
|---|---|
| Operating System | Ubuntu 24.04 LTS |
| Python | 3.12.3 |
| TensorFlow CPU | 2.17.0 |
| Keras | 3.15.1 |
| NumPy | 1.26.4 |
| Pillow | 13.0.0 |
| Matplotlib | 3.11.2 |
| Albumentations | 1.4.18 |

## Dataset

A synthetic dataset was generated using Python and Pillow.

| Dataset | Circles (Class A) | Squares (Class B) | Total |
|---|---:|---:|---:|
| Training | 200 | 200 | 400 |
| Testing | 50 | 50 | 100 |
| **Total** | **250** | **250** | **500** |

Image dimensions: 150 × 150 pixels (RGB).

The dataset is generated locally using `gen_data.py` and is not stored in Git.

## Image Augmentation Techniques

Three Keras preprocessing layers were used:

1. **Random Rotation:** Rotates images by up to 40 degrees.
2. **Random Flip:** Applies horizontal and vertical flips.
3. **Random Translation:** Shifts images horizontally and vertically by up to 20%.

The transformations are applied during training.

## Model Architecture

Both CNN models use the same architecture:

- Input layer: 150 × 150 × 3
- Rescaling layer: Normalizes pixel values to 0–1.
- Convolutional layer: 32 filters, 3 × 3 kernel.
- MaxPooling layer: 2 × 2.
- Flatten layer.
- Dense layer: 128 neurons with ReLU activation.
- Output layer: 1 neuron with sigmoid activation.

**Training configuration:**
- Optimizer: Adam
- Loss: Binary Crossentropy
- Metric: Accuracy
- Epochs: 3
- Batch size: 32

## Results

The following results were obtained from the test dataset:

| Model | Test Accuracy |
|---|---:|
| Without Augmentation | 64.00% |
| With Augmentation | 58.00% |

The baseline model achieved 64% test accuracy, while the augmented model achieved 58%.

In this experiment, augmentation did not improve test accuracy. The small dataset and limited training duration may have affected the augmented model's performance.

## Visualization

The `aug.png` file contains one original image and five randomly augmented versions.

## Project Files

| File / Directory | Description |
|---|---|
| `gen_data.py` | Generates synthetic training and testing images |
| `aug.py` | Applies and visualizes image augmentation |
| `train.py` | Trains and evaluates both CNN models |
| `requirements.txt` | Python package dependencies |
| `aug.png` | Augmentation visualization |
| `.gitignore` | Excludes virtual environment, cache, and generated datasets |
| `README.md` | Project documentation |

## How to Run

Clone the repository and enter the lab directory.

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate the dataset:

```bash
python gen_data.py
```

Run image augmentation:

```bash
python aug.py
```

Train and compare CNN models:

```bash
TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 OMP_NUM_THREADS=2 python train.py
```

## Conclusion

This lab demonstrated how image augmentation can introduce variations into training images using rotation, flipping, and translation.

Two CNN models were trained and evaluated using the same test dataset. The baseline model achieved 64% accuracy, while the augmented model achieved 58%.

The experiment provided practical experience with image preprocessing, augmentation pipelines, batch generation, CNN training, and model evaluation.

---

**Project:** Deep Learning – Al Razzaq Part 1
**Lab:** 21 – Data Augmentation for Images
