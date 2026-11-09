import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# Set a fixed random seed for reproducible results
np.random.seed(42)

# Load the Diabetes dataset
X_all, y_all = load_diabetes(return_X_y=True)

# Select 50 random samples
idx = np.random.RandomState(42).choice(len(y_all), 50, replace=False)

# Select BMI feature only
X = X_all[idx][:, [2]]
y = y_all[idx]

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.5, random_state=42
)
