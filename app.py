import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Load model, scaler, and label encoder
model = joblib.load('improved_models/model_random_forest_(smote).pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')
feature_columns = ['Soil_pH', 'Organic_Carbon', 'Clay_Content', 'Sand_Content', 'Silt_Content', 
                   'EC', 'Clay_Sand_Ratio', 'Texture_Sum', 'pH_EC_Interaction', 'Organic_Texture_Ratio']

# Prediction pipeline function
def predict_soil_type(new_data, model, scaler, label_encoder, feature_columns):
    try:
        if isinstance(new_data, dict):
            new_data = pd.DataFrame([new_data])
        required_cols = ['Soil_pH', 'Organic_Carbon', 'Clay_Content', 'Sand_Content', 'Silt_Content', 'EC']
        if not all(col in new_data.columns for col in required_cols):
            raise ValueError(f"Input data must contain columns: {required_cols}")
        # Validate input ranges
        if not (new_data['Soil_pH'].between(4, 9)).all():
            raise ValueError("Soil_pH must be between 4 and 9")
        if not (new_data['Organic_Carbon'].between(0, 10)).all():
            raise ValueError("Organic_Carbon must be between 0 and 10")
        if not (new_data[['Clay_Content', 'Sand_Content', 'Silt_Content']].between(0, 100)).all().all():
            raise ValueError("Clay/Sand/Silt_Content must be between 0 and 100")
        if not (new_data['EC'].between(0, 5)).all():
            raise ValueError("EC must be between 0 and 5")
        # Create engineered features
        new_data['Clay_Sand_Ratio'] = new_data['Clay_Content'] / (new_data['Sand_Content'] + 1e-6)
        new_data['Texture_Sum'] = new_data['Clay_Content'] + new_data['Sand_Content'] + new_data['Silt_Content']
        new_data['pH_EC_Interaction'] = new_data['Soil_pH'] * new_data['EC']
        new_data['Organic_Texture_Ratio'] = new_data['Organic_Carbon'] / (new_data['Texture_Sum'] + 1e-6)
        X_new = new_data[feature_columns]
        X_new_scaled = scaler.transform(X_new)
        predictions_encoded = model.predict(X_new_scaled)
        predictions = label_encoder.inverse_transform(predictions_encoded)
        return predictions.tolist()
    except Exception as e:
        raise ValueError(f"Prediction error: {str(e)}")

# API endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Convert JSON to DataFrame
        if isinstance(data, list):
            input_data = pd.DataFrame(data)
        else:
            input_data = pd.DataFrame([data])
        
        # Make predictions
        predictions = predict_soil_type(input_data, model, scaler, label_encoder, feature_columns)
        
        return jsonify({
            'predictions': predictions,
            'status': 'success'
        }), 200
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'error': f"Server error: {str(e)}"}), 500

# Root endpoint for API status
@app.route('/')
def status():
    return jsonify({
        'status': 'Soil Type Prediction API is running',
        'version': '1.0.0',
        'model': 'Random Forest (SMOTE)'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use PORT env variable for cloud deployment
    app.run(host='0.0.0.0', port=port, debug=False)