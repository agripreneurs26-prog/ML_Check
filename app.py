from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# ✅ Absolute base path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ✅ File paths
MODEL_PATH = os.path.join(BASE_DIR, 'optimized_xgboost.joblib')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.joblib')

# ✅ Load model & scaler safely
try:
    print("📂 Current directory:", BASE_DIR)
    print("📄 Files in directory:", os.listdir(BASE_DIR))

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    print("✅ Model and scaler loaded successfully")

except Exception as e:
    print("❌ Error loading model/scaler:", e)
    model = None
    scaler = None


# ✅ Expected input columns
expected_columns = [
    'gender', 'region', 'highest_education', 'imd_band', 'age_band',
    'num_of_prev_attempts', 'studied_credits', 'score'
]

# ✅ Columns to scale
features_to_scale = ['num_of_prev_attempts', 'studied_credits', 'score']


# ✅ Health check route (Render needs this)
@app.route('/')
def home():
    return jsonify({
        "status": "success",
        "message": "🚀 API is running successfully!"
    })


# ✅ Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 🔴 Check model loaded
        if model is None or scaler is None:
            return jsonify({
                'status': 'error',
                'message': 'Model or scaler not loaded'
            })

        data = request.get_json()

        # 🔴 Handle empty request
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No input data provided'
            })

        # 🔴 Validate input fields
        for col in expected_columns:
            if col not in data:
                return jsonify({
                    'status': 'error',
                    'message': f'Missing field: {col}'
                })

        # ✅ Convert to DataFrame
        input_df = pd.DataFrame([data])
        input_df = input_df[expected_columns]

        # ✅ Convert numeric safely
        input_df[features_to_scale] = input_df[features_to_scale].apply(pd.to_numeric, errors='coerce')

        # 🔴 Check for invalid numeric values
        if input_df[features_to_scale].isnull().any().any():
            return jsonify({
                'status': 'error',
                'message': 'Invalid numeric values provided'
            })

        # ✅ Scale features
        input_df[features_to_scale] = scaler.transform(input_df[features_to_scale])

        # ✅ Prediction
        prediction = model.predict(input_df)

        return jsonify({
            'status': 'success',
            'prediction': int(prediction[0])
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })


# ✅ Required for Render (IMPORTANT)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
