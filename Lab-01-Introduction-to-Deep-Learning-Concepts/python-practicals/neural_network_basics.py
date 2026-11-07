"""
Lab 1: Introduction to Deep Learning Concepts
Practical: A simple neural network learning the XOR problem.
"""

import numpy as np

# 1. Define the XOR training data
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

y = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=float)

# 2. Activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

# 3. Initialize network weights and biases
np.random.seed(7)

input_neurons = 2
hidden_neurons = 4
output_neurons = 1

W1 = np.random.randn(input_neurons, hidden_neurons)
b1 = np.zeros((1, hidden_neurons))

W2 = np.random.randn(hidden_neurons, output_neurons)
b2 = np.zeros((1, output_neurons))

learning_rate = 0.8
epochs = 10000

# 4. Train the neural network
for epoch in range(epochs):

    # Forward propagation
    hidden_input = np.dot(X, W1) + b1
    hidden_output = np.tanh(hidden_input)

    final_input = np.dot(hidden_output, W2) + b2
    predictions = sigmoid(final_input)

    # Calculate error
    error = predictions - y
    loss = np.mean(error ** 2)

    # Backpropagation
    output_gradient = (
        2 * error / len(X)
        * sigmoid_derivative(predictions)
    )

    hidden_gradient = (
        np.dot(output_gradient, W2.T)
        * (1 - hidden_output ** 2)
    )

    # Update weights and biases
    W2 -= learning_rate * np.dot(hidden_output.T, output_gradient)
    b2 -= learning_rate * np.sum(output_gradient, axis=0, keepdims=True)

    W1 -= learning_rate * np.dot(X.T, hidden_gradient)
    b1 -= learning_rate * np.sum(hidden_gradient, axis=0, keepdims=True)

    if (epoch + 1) % 2000 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss:.6f}")

# 5. Test the trained network
print("\nXOR Neural Network Predictions:")
print("Input -> Predicted Output")

for inputs, prediction in zip(X, predictions):
    result = int(prediction[0] >= 0.5)
    print(f"{inputs.astype(int)} -> {result} (confidence: {prediction[0]:.4f})")

print("\nPractical completed successfully!")
