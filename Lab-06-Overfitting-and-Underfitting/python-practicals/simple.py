from data import *

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse

# Compare a simple model with a complex model
for d in (1, 10):

    # Build the machine learning pipeline
    m = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(d, include_bias=False),
        StandardScaler(),
        LinearRegression()
    )

    # Train the model
    m.fit(X_train, y_train)

    # Calculate training and validation errors
    train_error = mse(y_train, m.predict(X_train))
    val_error = mse(y_val, m.predict(X_val))

    print(
        "Degree:", d,
        "| Training MSE:", round(train_error, 1),
        "| Validation MSE:", round(val_error, 1)
    )
