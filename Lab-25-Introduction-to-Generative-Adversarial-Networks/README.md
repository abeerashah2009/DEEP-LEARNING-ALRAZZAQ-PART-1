LAB 25: INTRODUCTION TO GENERATIVE ADVERSARIAL NETWORKS (GANs)
================================================================

1. OBJECTIVE
------------

The objective of this lab is to understand Generative Adversarial
Networks (GANs).

In this lab, we build two neural networks:
- Generator
- Discriminator

We train both networks using the MNIST handwritten digit dataset.
The Generator creates new synthetic images from random noise.


2. WHAT IS A GAN?
-----------------

GAN stands for Generative Adversarial Network.

A GAN contains two neural networks that compete with each other.

A) GENERATOR
The Generator creates fake images using random noise.

B) DISCRIMINATOR
The Discriminator checks whether an image is real or generated.

The Generator tries to fool the Discriminator.
The Discriminator tries to identify fake images correctly.


3. TECHNOLOGIES USED
--------------------

- Python 3.12
- TensorFlow CPU 2.18.0
- Keras 3.15.1
- NumPy 2.0.2
- Matplotlib 3.11.2
- Ubuntu Linux


4. GENERATOR ARCHITECTURE
-------------------------

The Generator receives random noise with 100 values.

Layers:
- Input: 100 values
- Dense: 128 neurons, ReLU
- Dense: 256 neurons, ReLU
- Dense: 512 neurons, ReLU
- Dense: 784 neurons, Tanh
- Reshape: 28 x 28 x 1

Total parameters: 579,728

Output: 28 x 28 grayscale image.


5. DISCRIMINATOR ARCHITECTURE
-----------------------------

The Discriminator receives an image and predicts
whether it is real or generated.

Layers:
- Input: 28 x 28 x 1
- Flatten
- Dense: 512 neurons, ReLU
- Dense: 256 neurons, ReLU
- Dense: 1 neuron, Sigmoid

Total parameters: 533,505

Output: Probability of an image being real.


6. DATASET: MNIST
-----------------

MNIST is a dataset of handwritten digits.

Dataset information:
- 60,000 training images
- Image size: 28 x 28 pixels
- Grayscale images
- Pixel values normalized between -1 and 1

Python code:

(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()

x_train = (x_train.astype('float32') - 127.5) / 127.5

x_train = np.expand_dims(x_train, axis=-1)

Expected output:

(60000, 28, 28, 1)


7. TRAINING CONFIGURATION
-------------------------

- Training iterations: 2,000
- Batch size: 64
- Optimizer: Adam
- Learning rate: 0.0002
- Beta 1: 0.5
- Loss function: Binary Crossentropy
- Real image labels: 0.9
- Fake image labels: 0

TensorFlow GradientTape is used to update the
Generator and Discriminator separately.


8. TRAINING RESULTS
-------------------

Selected results from the completed training:

Iteration 0:
Discriminator Loss = 1.864
Discriminator Accuracy = 0.49
Generator Loss = 0.829

Iteration 400:
Discriminator Loss = 0.749
Discriminator Accuracy = 1.00
Generator Loss = 2.324

Iteration 800:
Discriminator Loss = 0.648
Discriminator Accuracy = 0.96
Generator Loss = 3.086

Iteration 1200:
Discriminator Loss = 0.804
Discriminator Accuracy = 0.88
Generator Loss = 2.494

Iteration 1600:
Discriminator Loss = 0.697
Discriminator Accuracy = 0.97
Generator Loss = 2.515

Iteration 1800:
Discriminator Loss = 0.854
Discriminator Accuracy = 0.80
Generator Loss = 1.324


9. GENERATING SYNTHETIC IMAGES
------------------------------

After training, the Generator receives random noise
and creates 10 synthetic images.

Generated image shape:

(10, 28, 28, 1)

The images are saved in:

gan.png

This file contains a grid of 10 generated grayscale images.


10. PROJECT FILES
-----------------

gan.py
Main Python program containing the GAN implementation.

gan.png
Generated image visualization.

README.md
Lab notes and documentation.

requirements.txt
Required Python packages.


11. HOW TO RUN THE LAB
----------------------

Step 1: Create virtual environment

python3 -m venv venv

Step 2: Activate environment

source venv/bin/activate

Step 3: Install packages

pip install -r requirements.txt

Step 4: Run the program

python gan.py


12. LEARNING OUTCOMES
---------------------

After completing this lab, I learned:

- What Generative Adversarial Networks are.
- Difference between Generator and Discriminator.
- How to build GAN models using Keras.
- How to load and preprocess MNIST data.
- How to train two neural networks.
- How to generate synthetic images.
- How to save generated images using Matplotlib.


13. CONCLUSION
--------------

In this lab, I implemented a Generative Adversarial Network
using TensorFlow and Keras.

I built and trained a Generator and a Discriminator
using the MNIST dataset.

The Generator created synthetic grayscale images,
and the Discriminator learned to identify real and
generated images.

The generated output was saved as gan.png.

LAB 25 COMPLETED
================================================================
