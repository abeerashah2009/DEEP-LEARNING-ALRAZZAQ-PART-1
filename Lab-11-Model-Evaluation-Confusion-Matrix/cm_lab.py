import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

# Task 1: Load and split the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Task 2: Train the Decision Tree model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Generate predictions
y_pred = clf.predict(X_test)
print("\nFirst 10 predictions:", y_pred[:10])

# Task 3: Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Task 4: Visualize and save the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Decision Tree Classifier")
plt.tight_layout()
plt.savefig("cm.png", dpi=100)
plt.close()

print("\nConfusion matrix image saved: cm.png")

# Task 5: Classification report
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names
))

# Task 6: Calculate metrics manually
tp = np.diag(cm)
fp = cm.sum(axis=0) - tp
fn = cm.sum(axis=1) - tp

precision = tp / (tp + fp)
recall = tp / (tp + fn)

f1_score = 2 * precision * recall / (precision + recall)

print("\nManual Metrics:")
for i, name in enumerate(iris.target_names):
    print(
        f"{name}: "
        f"Precision={precision[i]:.2f}, "
        f"Recall={recall[i]:.2f}, "
        f"F1-Score={f1_score[i]:.2f}"
    )

manual_accuracy = tp.sum() / cm.sum()
model_accuracy = accuracy_score(y_test, y_pred)

print("\nManual Accuracy:", manual_accuracy)
print("Sklearn Accuracy:", model_accuracy)
print("Accuracy values match:", np.isclose(manual_accuracy, model_accuracy))

# Task 7: Interpret classification errors
print("\nError Analysis:")
print("Total incorrect predictions:", np.sum(cm) - np.trace(cm))

if np.sum(cm) == np.trace(cm):
    print("No classification errors were found in this test set.")
else:
    print("Some samples were incorrectly classified.")
    print("Check the off-diagonal values in the confusion matrix.")
