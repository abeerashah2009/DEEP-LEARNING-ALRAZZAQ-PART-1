from data import *

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error as mse

# Compare normal regression with Ridge regularization
models = [
    ("plain", LinearRegression()),
    ("ridge", Ridge(alpha=10.0))
]

for name, reg in models:

    m = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(10, include_bias=False),
        StandardScaler(),
        reg
    )

    # Train the model
    m.fit(X_train, y_train)

    # Calculate validation error
    val_error = mse(y_val, m.predict(X_val))

    print(name, "validation MSE:", round(val_error, 1))
