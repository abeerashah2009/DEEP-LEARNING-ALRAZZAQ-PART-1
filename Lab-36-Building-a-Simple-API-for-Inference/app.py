import joblib
from flask import Flask, jsonify, request

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return "Welcome to the Simple API for Inference!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True, silent=True)
    if not isinstance(data, dict) or 'input' not in data:
        return jsonify({'error': "JSON body with an 'input' list is required"}), 400
    try:
        prediction = model.predict([data['input']])
    except (ValueError, TypeError):
        return jsonify({'error': 'input must be a list of 4 numbers'}), 400
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
