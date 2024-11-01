from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained air quality model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data from HTML input fields
        data = request.form
        features = np.array([[
            float(data['CO_GT']),
            float(data['PT08_S1_CO']),
            float(data['C6H6_GT']),
            float(data['PT08_S2_NMHC']),
            float(data['NOx_GT']),
            float(data['PT08_S3_NOx']),
            float(data['NO2_GT']),
            float(data['PT08_S4_NO2']),
            float(data['PT08_S5_O3']),
            float(data['T']),
            float(data['RH']),
            float(data['AH'])
        ]])

        # Make prediction using the loaded model
        prediction = model.predict(features)[0]

        # Map prediction to labels
        if prediction < 1:
            label = 'Very Low'
        elif 1 <= prediction < 3:
            label = 'Low'
        elif 3 <= prediction < 7:
            label = 'High'
        else:
            label = 'Very High'

        result = f'Predicted Air Quality Level: {label}'
        return render_template('index.html', prediction_text=result)

    except Exception as e:
        print("Prediction Error:", e)  # Log error to console
        return render_template('index.html', prediction_text="Error occurred while predicting.")

@app.route('/api/predict', methods=['POST'])
def api_predict():
    # Extract JSON data from API request
    data = request.get_json()
    features = np.array([[ 
        data['CO_GT'],
        data['PT08_S1_CO'],
        data['C6H6_GT'],
        data['PT08_S2_NMHC'],
        data['NOx_GT'],
        data['PT08_S3_NOx'],
        data['NO2_GT'],
        data['PT08_S4_NO2'],
        data['PT08_S5_O3'],
        data['T'],
        data['RH'],
        data['AH']
    ]], dtype=float)

    # Make prediction using the loaded model
    prediction = model.predict(features)[0]

    # Map prediction to labels
    if prediction < 1:
        label = 'Very Low'
    elif 1 <= prediction < 3:
        label = 'Low'
    elif 3 <= prediction < 7:
        label = 'High'
    else:
        label = 'Very High'

    result = {'predicted_air_quality': label}
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
