
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist


# Create the evidence directory
EVIDENCE_DIR = Path(__file__).resolve().parent.parent / "evidence"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)


# Step 1: Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# Step 2: Display dataset shapes
print("Train images shape:", train_images.shape)
print("Train labels shape:", train_labels.shape)
print("Test images shape:", test_images.shape)
print("Test labels shape:", test_labels.shape)


# Step 3: Normalize pixel values between 0 and 1
train_images = train_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0


# Step 4: Verify normalization
print("Max pixel value after normalization:", train_images.max())
print("Min pixel value after normalization:", train_images.min())


# Step 5: Visualize sample handwritten digits
def visualize_samples(images, labels):
    plt.figure(figsize=(10, 5))

    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i], cmap="gray")
        plt.title(f"Label: {labels[i]}")
        plt.axis("off")

    plt.tight_layout()

    output_file = EVIDENCE_DIR / "samples.png"
    plt.savefig(output_file)
    plt.close()

    print("Visualization saved to:", output_file)


# Step 6: Display the first 10 labels
visualize_samples(train_images, train_labels)

print("First 10 labels:", train_labels[:10].tolist())
