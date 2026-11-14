# Lab 26 - Building a Variational Autoencoder (VAE)

## Objective
The objective of this lab is to understand Variational Autoencoders (VAEs), build an encoder and decoder, train the model on MNIST, and visualize the latent space.

## What is a VAE?
A Variational Autoencoder is a type of generative neural network.

It learns a compressed representation of input data and can generate new data samples.

## VAE Architecture

The VAE contains three main parts:

1. Encoder
   - Takes an input image of 784 values.
   - Uses Dense layers with 512 and 256 neurons.
   - Produces z_mean and z_log_var.
   - Creates a 2-dimensional latent representation.

2. Sampling Layer
   - Uses the reparameterization trick.
   - Samples a latent vector using z_mean and z_log_var.
   - Calculates the KL-divergence loss.

3. Decoder
   - Takes a 2-dimensional latent vector.
   - Uses Dense layers with 256 and 512 neurons.
   - Produces 784 output values.
   - Uses sigmoid activation to reconstruct the image.

## Dataset
The MNIST handwritten digits dataset was used.

- Training images: 60,000
- Testing images: 10,000
- Image size: 28 x 28 pixels
- Input features: 784

## Model Training
- Optimizer: Adam
- Batch size: 128
- Epochs: 50
- Latent dimension: 2

The model uses two loss components:

1. Reconstruction Loss
   Measures the difference between the original image and reconstructed image.

2. KL-Divergence Loss
   Helps organize the latent space into a distribution.

The total VAE objective combines reconstruction loss and KL-divergence loss.

A Keras 3-compatible custom Sampling layer was used to add the KL-divergence loss during training.

## Training Results

- Initial training loss: 48.6436
- Final training loss: 33.5246
- Final validation loss: 34.5577
- Completed epochs: 50

The training loss decreased during training.

## Latent Space Visualization

The trained decoder generated a 15 x 15 grid of digits using latent values between -3 and 3.

Output image:

vae_latent_grid.png

## Files

- vae.py: VAE model, training, and visualization code.
- vae_latent_grid.png: Generated latent-space visualization.
- requirements.txt: Python dependencies.
- README.md: Lab documentation.

## Conclusion

In this lab, I built and trained a Variational Autoencoder using TensorFlow and Keras.

I learned about encoder and decoder architecture, latent variables, the reparameterization trick, reconstruction loss, and KL-divergence.

The model successfully completed 50 training epochs and generated a latent-space visualization using MNIST.

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- MNIST Dataset
