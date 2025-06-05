import pandas as pd
import numpy as np
import joblib
import requests
import json

# Load new patient data
new_data = pd.read_csv("C:\\Users\\A K Pandey\\Downloads\\Project\\new_patient_data.csv")

# Define columns to scale and ordered features

# Load feature names
feature_names = joblib.load('models/feature_names.pkl')
# Load scaler
scaler, columns_to_scale = joblib.load('models/scaler.pkl')

# Add missing columns for scaling if not present
for col in feature_names:
    if col not in new_data.columns:
        new_data[col] = 0.0  # Add missing columns with default value
asa_mapping = {'ASA_one': 1, 'ASA_two': 2, 'ASA_three': 3, 'ASA_four': 4, 'ASA-E': 5}
if 'ASAclassification' in new_data.columns:
    new_data['ASAclassification'] = new_data['ASAclassification'].map(asa_mapping)

new_data = new_data[feature_names]

# Fill missing values and scale numeric columns
new_data = new_data.fillna(0.0)
new_data[columns_to_scale] = scaler.transform(new_data[columns_to_scale])

# API endpoint
url = 'http://127.0.0.1:5000/predict'
results = []

for idx, row in new_data.iterrows():
    # Replace NaN with None for JSON serialization
    row_cleaned = row.apply(lambda x: None if pd.isna(x) else x)
    patient_json = [row_cleaned.to_dict()]

    try:
        response = requests.post(url, json=patient_json)
        response.raise_for_status()
        result = response.json()

        prediction = result.get("prediction", [None])[0]
        probability = result.get("raw_probability", [None])[0]
        std_dev = result.get("std_deviation", [None])[0]
        entropy = result.get("entropy", [None])[0]
        calibrated_prob = result.get("calibrated_probability", [None])[0]
        print(f"📤 Sending patient {idx+1} to the server")
        print(f"Patient {idx+1}: 📊 Probability: {probability:.4f} | ✅ Prediction: {prediction} | "
              f"📉 Std Dev: {std_dev:.4f} | 🧠 Entropy: {entropy:.4f} | "
              f"🎯 Calibrated Prob: {calibrated_prob if calibrated_prob is not None else 'N/A'}")

        results.append({
            "Patient_ID": idx + 1,
            "Predicted_Probability": probability,
            "Predicted_Class": prediction,
            "Standard_Deviation": std_dev,
            "Entropy": entropy,
            "Calibrated_Probability": calibrated_prob
        })

    except requests.exceptions.HTTPError as http_err:
        print(f"❌ Patient {idx+1}: HTTP error: {http_err}")
    except Exception as err:
        print(f"❌ Patient {idx+1}: Other error: {err}")

# Save local copy (optional, renamed to avoid conflict)
try:
    results_df = pd.DataFrame(results)
    results_df.to_csv("C:\\Users\\A K Pandey\\Downloads\\Project\\new_patient_predictions_client.csv", index=False)
    print("✅ Local copy saved as 'new_patient_predictions_client.csv'")
except PermissionError:
    print("❌ Local CSV file open. Please close and retry.")
except Exception as e:
    print(f"❌ Failed to save predictions: {e}")