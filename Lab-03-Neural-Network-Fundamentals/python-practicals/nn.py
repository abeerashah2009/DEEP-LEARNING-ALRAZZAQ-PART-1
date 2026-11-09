import numpy as np

rng = np.random.default_rng(42)

# Input layer: one sample with 3 features
input_features = np.array([[1, 2, 3]])

# Layer sizes
hidden_layer_size = 4
output_layer_size = 2

print("Input shape:", input_features.shape)

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Process input through the hidden neurons
weights_input_hidden = rng.random((3, hidden_layer_size))
neuron_input = np.dot(input_features, weights_input_hidden)
neuron_output = sigmoid(neuron_input)

print("Hidden sigmoid output:", neuron_output)

# ReLU and Tanh activation functions
def relu(x):
    return np.maximum(0, x)

print("ReLU output:", relu(neuron_input))
print("Tanh output:", np.tanh(neuron_input))

# Feedforward process
def feedforward(input_data, w_input_hidden, w_hidden_output):
    # Input to hidden layer
    hidden_input = np.dot(input_data, w_input_hidden)
    hidden_output = sigmoid(hidden_input)

    # Hidden to output layer
    final_input = np.dot(hidden_output, w_hidden_output)
    return sigmoid(final_input)

weights_hidden_output = rng.random((hidden_layer_size, output_layer_size))

output = feedforward(input_features, weights_input_hidden, weights_hidden_output)

print("Feedforward Output:", output)
print("Output shape:", output.shape)
print("All values in 0..1:", bool(((output > 0) & (output < 1)).all()))
