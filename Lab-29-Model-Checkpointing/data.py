from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Load the handwritten digits dataset
X, y = load_digits(return_X_y=True)

# Normalize pixel values from 0-16 to 0-1
X = X / 16.0

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
