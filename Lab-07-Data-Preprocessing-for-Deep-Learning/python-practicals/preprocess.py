# ==========================================
# Lab 07: Data Preprocessing for Deep Learning
# ==========================================

# ---- Step 1: Imports ----
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split


# ---- Step 2: Normalize a small sample with MinMaxScaler ----
# MinMaxScaler rescales each column to the [0, 1] range.

data = np.array([[10, 2.7, 3.6],
                 [-100, 5, -2],
                 [120, 20, 40]], dtype=np.float64)

normalized_sample = MinMaxScaler().fit_transform(data)

print("Normalized Sample:\n", normalized_sample.round(3))
# ---- Step 3: Standardize the same sample with StandardScaler ----
# StandardScaler gives each column mean 0 and standard deviation 1.

standardized_sample = StandardScaler().fit_transform(data)

print("Standardized Sample:\n", standardized_sample.round(3))

print("Sample column means:", standardized_sample.mean(axis=0).round(3))

print("Sample column stds:", standardized_sample.std(axis=0).round(3))
# ---- Step 4: Create features and target ----
rng = np.random.default_rng(42)
X = rng.random((100, 5))
y = rng.integers(0, 2, 100)


# ---- Step 5: Split FIRST (70% train / 30% test) ----
# random_state makes the split reproducible.
# stratify=y keeps the class balance similar.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("Training Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)
# ---- Step 6: Normalize; fit on train only, then transform test ----
# Fitting on the test set too would leak information (data leakage).

minmax = MinMaxScaler().fit(X_train)

X_train_norm = minmax.transform(X_train)
X_test_norm = minmax.transform(X_test)

print("Normalized train min/max:",
      X_train_norm.min().round(3),
      X_train_norm.max().round(3))

print("Test normalized min/max:",
      X_test_norm.min().round(3),
      X_test_norm.max().round(3))
# ---- Step 7: Standardize; fit on train only, then transform test ----

std = StandardScaler().fit(X_train)

X_train_std = std.transform(X_train)
X_test_std = std.transform(X_test)

print("Standardized train mean:", X_train_std.mean().round(3))
print("Standardized train std:", X_train_std.std().round(3))
