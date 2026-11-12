import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Load training history
with open("history.json", "r") as f:
    h = json.load(f)

# Plot loss curves
plt.figure()
plt.plot(h["loss"], label="Train Loss")
plt.plot(h["val_loss"], label="Validation Loss")
plt.title("Loss Curves")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss.png")

# Plot accuracy curves
plt.figure()
plt.plot(h["accuracy"], label="Train Accuracy")
plt.plot(h["val_accuracy"], label="Validation Accuracy")
plt.title("Accuracy Curves")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("accuracy.png")

print("Both training graphs saved successfully!")
