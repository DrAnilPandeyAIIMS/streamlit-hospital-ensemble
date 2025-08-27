# 🏥 Mortality Prediction App for Surgeons

This is a web-based application that helps surgeons assess the risk of patient mortality using an ensemble of deep learning models.

### ✅ How to Use

1. Go to the website (the link will be shared by your team).
2. Click **"Upload CSV"** and select a file with patient data.
3. Click **"Predict"**.
4. You will see the risk prediction and confidence (uncertainty).
5. Download the result as CSV if needed.

### 📁 Input File Format

Your file must be in `.csv` format (spreadsheet) and contain specific columns (like age, vitals, lab results). Ask the technical team for a template.

### 🔍 Models Used

- Variational Autoencoder (VAE)
- Bayesian Neural Networks
- Calibrated Ensemble Model

### ⚙️ Developers / Technical Info

- App built with Streamlit
- TensorFlow models (.h5 and SavedModel)
- Python 3.8.19
- TFP version: 0.21
- TensorFlow version: 2.13

### 📦 Requirements

See `requirements.txt` for full dependencies.

