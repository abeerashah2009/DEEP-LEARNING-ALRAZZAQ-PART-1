import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from data import *

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse

# Test polynomial degrees from 1 to 10
degrees = list(range(1, 11))

tr = []
va = []

lines = ["degree  train_MSE  val_MSE  gap"]

# Train and evaluate each model
for d in degrees:

    m = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(d, include_bias=False),
        StandardScaler(),
        LinearRegression()
    )

    m.fit(X_train, y_train)

    train_error = mse(y_train, m.predict(X_train))
    val_error = mse(y_val, m.predict(X_val))

    tr.append(train_error)
    va.append(val_error)

    lines.append(
        f"{d:6d} {train_error:10.1f} {val_error:9.1f} {val_error - train_error:8.1f}"
    )

# Save results in a text file
open("table.txt", "w").write("\n".join(lines) + "\n")

print("\n".join(lines))

# Plot training and validation errors
plt.plot(degrees, tr, marker="o", label="Training MSE")
plt.plot(degrees, va, marker="o", label="Validation MSE")

plt.xlabel("Polynomial degree")
plt.ylabel("Mean Squared Error")
plt.yscale("log")
plt.legend()
plt.title("Training vs Validation Loss")

plt.savefig("../loss.png", dpi=100)

# Find the best degree
print("Best degree (lowest val MSE):", degrees[va.index(min(va))])

# Verify expected overfitting behaviour
assert tr[-1] < tr[0], "Training MSE should be lower at degree 10"
assert (va[-1] - tr[-1]) > (va[0] - tr[0]), "Gap should grow with degree"

print("Assertions passed")
