import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense

# Set random seeds for reproducible results
np.random.seed(42)
tf.keras.utils.set_random_seed(42)

# Model parameters
vocab_size = 1000
embedding_dim = 50
seq_len = 10

# Build the embedding model
model = Sequential([
    Input(shape=(seq_len,)),
    Embedding(input_dim=vocab_size, output_dim=embedding_dim),
    Flatten(),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Display model architecture
model.summary()
# Small text dataset
texts = [
    "This is an example sentence",
    "Another text data example"
]

y_data = np.array([0, 1])

# Convert text into integer token sequences
vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=seq_len
)

vectorizer.adapt(texts)

X_data = vectorizer(tf.constant(texts)).numpy()

# Display vocabulary and numerical data
print("\nVocabulary:")
print(vectorizer.get_vocabulary())

print("\nX_data:")
print(X_data)
# Train the model
print("\nTraining the model:")

model.fit(
    X_data,
    y_data,
    epochs=10,
    batch_size=2,
    verbose=2
)

# Display predictions
predictions = model.predict(X_data, verbose=0)

print("\nPredictions:")
print(predictions.ravel())
# Extract learned embeddings
embeddings = model.layers[0].get_weights()[0]

print("\nEmbedding matrix shape:")
print(embeddings.shape)

# Get the vocabulary
vocab = vectorizer.get_vocabulary()

# Find the embedding vector for the word 'example'
idx = vocab.index('example')

print("\nVector for 'example' (first 5 dimensions):")
print(embeddings[idx][:5])
# Visualize embeddings using t-SNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Number of words in our vocabulary
n_words = len(vocab)

# Reduce embedding dimensions from 50 to 2
tsne = TSNE(
    n_components=2,
    perplexity=3,
    random_state=42
)

reduced = tsne.fit_transform(embeddings[:n_words])

# Create the scatter plot
plt.figure(figsize=(10, 7))
plt.scatter(reduced[:, 0], reduced[:, 1])

# Add word labels
for i, w in enumerate(vocab):
    plt.annotate(w if w else '<pad>', reduced[i])

plt.title('t-SNE Visualization of Embeddings')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.savefig('tsne_embeddings.png')
plt.close()

print("\nReduced embedding shape:")
print(reduced.shape)

print("\nVisualization saved as tsne_embeddings.png")
