# Lab 36: Building a Simple API for Inference

## 1. Introduction

In this lab, I learned how to build a simple REST API using Python and Flask.

The API uses a trained Machine Learning model to make predictions from input data.

**Main purpose:** Send input data to the API and receive a prediction as a response.

## 2. Lab Objectives

- Understand the basics of REST API.
- Learn how to use Flask.
- Train a Machine Learning model.
- Save and load a trained model using Joblib.
- Create an API prediction endpoint.
- Send JSON data using POST requests.
- Test valid and invalid API requests.

## 3. Technologies Used

- Python 3
- Flask 3.0.3
- Scikit-learn 1.5.2
- Joblib 1.4.2
- NumPy
- Ubuntu Linux
- cURL

## 4. Step 1 – Create Virtual Environment

A virtual environment keeps project dependencies separate from the system Python packages.

Command:

```bash
python3 -m venv venv
source venv/bin/activate
```

After activation, `(venv)` appears in the terminal.

## 5. Step 2 – Install Required Libraries

Install Flask:

```bash
pip install Flask==3.0.3
```

Install Machine Learning libraries:

```bash
pip install scikit-learn==1.5.2 joblib==1.4.2 numpy
```

**Purpose of libraries:**

- Flask: Creates the web API.
- Scikit-learn: Trains the Machine Learning model.
- Joblib: Saves and loads the trained model.
- NumPy: Supports numerical operations.

## 6. Step 3 – Create the Training Script

Create the file:

```bash
nano train.py
```

The training script uses the Iris dataset and Logistic Regression algorithm.

```python
import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

X, y = load_iris(return_X_y=True)
model = LogisticRegression(max_iter=200)
model.fit(X, y)
joblib.dump(model, 'model.pkl')
print('Saved model.pkl, training accuracy:', round(model.score(X, y), 3))
```

Run the training script:

```bash
python train.py
```

**Observed output:**

```text
Saved model.pkl, training accuracy: 0.973
```

The trained model is saved in `model.pkl`.

## 7. Step 4 – Create Flask Application

Create the application:

```bash
nano app.py
```

The Flask application contains two endpoints:

- `/` – Displays a welcome message.
- `/predict` – Receives input data and returns a prediction.

The application loads the trained model using Joblib.

## 8. Step 5 – Start Flask Server

Command:

```bash
flask --app app run --port 5000 &
```

Server address:

```text
http://127.0.0.1:5000
```

The Flask server listens on port 5000.

## 9. Step 6 – Test Home Endpoint

Command:

```bash
curl http://127.0.0.1:5000/
```

**Observed response:**

```text
Welcome to the Simple API for Inference!
```

HTTP status: `200 OK`

This confirms that the home endpoint is working.

## 10. Step 7 – Create Sample JSON Input

Create the file:

```bash
nano sample_input.json
```

Input:

```json
{
    "input": [5.1, 3.5, 1.4, 0.2]
}
```

The input contains four measurements of an Iris flower.

## 11. Step 8 – Test Prediction Endpoint

Command:

```bash
curl -s -X POST http://127.0.0.1:5000/predict \
-H "Content-Type: application/json" \
-d @sample_input.json
```

**Observed response:**

```json
{"prediction":[0]}
```

Prediction `0` represents the Iris setosa class.

This confirms that the API successfully received the input and returned a Machine Learning prediction.

## 12. Step 9 – Test Invalid JSON

Command:

```bash
curl -s -w "\nHTTP Status: %{http_code}\n" \
-X POST http://127.0.0.1:5000/predict \
-H "Content-Type: application/json" \
-d 'not json'
```

**Observed response:**

```json
{"error":"JSON body with an 'input' list is required"}
```

HTTP status: `400`

This test confirms that the API handles invalid JSON requests.

## 13. Step 10 – Test Incorrect Input Length

Command:

```bash
curl -s -w "\nHTTP Status: %{http_code}\n" \
-X POST http://127.0.0.1:5000/predict \
-H "Content-Type: application/json" \
-d '{"input": [1, 2]}'
```

**Observed response:**

```json
{"error":"input must be a list of 4 numbers"}
```

HTTP status: `400`

The model requires four input values. It correctly rejects incomplete input.

## 14. Step 11 – Stop Flask Server

Find the Flask process:

```bash
ss -ltnp | grep ':5000'
```

Stop the process using its PID:

```bash
kill 14673
```

Verify that port 5000 is free:

```bash
ss -ltnp | grep ':5000'
```

Test the stopped server:

```bash
curl -s -o /dev/null -w "HTTP Status: %{http_code}\n" \
http://127.0.0.1:5000/
```

**Observed output:**

```text
HTTP Status: 000
```

This confirms that the Flask server has stopped.

## 15. Important Concepts

**REST API:** Allows applications to communicate through HTTP requests and responses.

**Flask:** A lightweight Python framework for creating web applications and APIs.

**POST Request:** Sends data to an API.

**JSON:** A format used to exchange data between applications.

**Inference:** Using a trained Machine Learning model to make predictions.

**Joblib:** A Python library used to save and load Machine Learning models.

**HTTP 200:** Request completed successfully.

**HTTP 400:** The request contains invalid input.

## 16. Lab Conclusion

In this lab, I successfully created a simple Machine Learning inference API using Flask.

I trained a Logistic Regression model using the Iris dataset and saved it using Joblib.

I connected the trained model with the Flask application and tested the prediction endpoint using cURL.

I also tested invalid JSON and incorrect input length. The API returned the expected error responses.

**Result:** Successfully built and tested a simple Flask API for Machine Learning inference.
