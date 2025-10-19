# streamlit_app_2.py
import os
import json
import warnings
import sys
import threading
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import tensorflow_probability as tfp
import joblib
import gdown
import gspread
from sklearn.isotonic import IsotonicRegression
from google.oauth2.service_account import Credentials

warnings.filterwarnings("ignore")
sys.path.append(".")

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Hospital Predictor - Mobile",
                   layout="centered",
                   initial_sidebar_state="collapsed")

st.title("🏥 Hospital Mortality Predictor (Mobile)")

# -----------------------------
# Model registry (Google Drive)
# -----------------------------
MODEL_FILES = {
    "vae_model": {
        "id": "1GXrJ4GvXOZ4ZzjqQQzfwlyWi9IkSswYe",
        "path": "models/vae_model.h5",
    },
    "model_1_custom": {
        "id": "1--Jlh2Zc7tCP80coi8pMl3QSrnK5TVHp",   # <--- your MyDrive model ID
        "path": "models/model_1_custom.h5",
    },
    "model_2_probabilistic": {
        "id": "1ug_BZlcHXwIiOdmC-fnI9SX-ye_ftrad",
        "path": "models/model_2_probabilistic.h5",
    },
}

# -----------------------------
# Download helper
# -----------------------------
def download_model_if_needed(model_key):
    info = MODEL_FILES[model_key]
    path = info["path"]
    if os.path.exists(path):
        return path
    try:
        gdown.download(id=info["id"], output=path, quiet=False)
        if os.path.exists(path):
            return path
        else:
            raise FileNotFoundError(f"Download failed: {path}")
    except Exception as e:
        st.error(f"❌ Could not download {model_key}: {e}")
        st.stop()

# -----------------------------
# Custom classes / objects
# -----------------------------
# -----------------------------
# Custom Layers / Losses
# -----------------------------
class DenseFlipoutLayer(tf.keras.layers.Layer):
    """Custom wrapper for Flipout dense layer."""
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = activation
    def build(self, input_shape):
        self.dense_flipout = tfp.layers.DenseFlipout(
            units=self.units, activation=self.activation
        )
    def call(self, inputs):
        return self.dense_flipout(inputs)
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

# ✅ Standard negative log-likelihood for Bernoulli outputs
def negative_log_likelihood_bernoulli(y_true, y_pred):
    return -tf.reduce_mean(
        y_true * tf.math.log(y_pred + 1e-9) +
        (1 - y_true) * tf.math.log(1 - y_pred + 1e-9)
    )

# ✅ Generic NLL for probabilistic (distribution) outputs
def negative_log_likelihood(y_true, y_pred_dist):
    """
    y_pred_dist is a Distribution object (e.g., Bernoulli, Normal, etc.)
    This computes the mean negative log-likelihood.
    """
    return -y_pred_dist.log_prob(y_true)

# ✅ Register both custom losses for model loading


# Register only necessary objects for Flipout models
custom_objects = {
    "DenseFlipoutLayer": DenseFlipoutLayer,
    "DenseFlipout": tfp.layers.DenseFlipout,
    "DistributionLambda": tfp.layers.DistributionLambda,
    "negative_log_likelihood": negative_log_likelihood,
    "negative_log_likelihood_bernoulli": negative_log_likelihood_bernoulli,
}

# -----------------------------
# Model loader
# -----------------------------
# -----------------------------
# Model loader
# -----------------------------
@st.cache_resource(hash_funcs={dict: lambda _: None})
def load_model_from_drive(model_key):
    path = download_model_if_needed(model_key)
    return tf.keras.models.load_model(path, custom_objects=custom_objects)

_loaded_models = {}
_loaded_lock = threading.Lock()

def get_model(model_key):
    with _loaded_lock:
        if model_key not in _loaded_models:
            _loaded_models[model_key] = load_model_from_drive(model_key)
    return _loaded_models[model_key]

vae_model = get_model("vae_model")
model_1 = get_model("model_1_custom")
model_2 = get_model("model_2_probabilistic")

# -----------------------------
# Ensemble predictions
# -----------------------------
# -----------------------------
# Ensemble predictions
# -----------------------------
def ensemble_models_predict_all(input_array, n_forward_passes=10):
    input_tensor = tf.convert_to_tensor(input_array, dtype=tf.float32)
    all_model_probs = []
    for model in [vae_model, model_1, model_2]:
        model_probs = []
        for _ in range(n_forward_passes):
            preds = model(input_tensor, training=True).numpy()
            if preds.ndim == 2 and preds.shape[1] == 1:
                preds = preds.flatten()
            model_probs.append(preds)
        all_model_probs.append(np.array(model_probs))
    return np.concatenate(all_model_probs, axis=0)


def calculate_entropy(probs):
    probs = np.clip(np.ravel(probs), 1e-9, 1-1e-9)
    return -(probs * np.log2(probs) + (1 - probs) * np.log2(1 - probs))

# -----------------------------
# Calibration
# -----------------------------
@st.cache_resource
def load_iso_reg():
    return joblib.load("models/iso_reg.pkl")

iso_reg = load_iso_reg()

try:
    with open("best_threshold.json", "r") as f:
        best_threshold = json.load(f)["best_threshold"]
except Exception as e:
    st.error(f"❌ Could not load best_threshold.json: {e}")
    st.stop()

# -----------------------------
# Load scaler + features
# -----------------------------
@st.cache_resource
def load_scaler_and_features():
    scaler_path = "models/scaler.pkl"
    features_path = "models/feature_names.pkl"

    obj = joblib.load(scaler_path)
    scaler = obj[0] if isinstance(obj, tuple) and hasattr(obj[0], "transform") else obj
    if not hasattr(scaler, "transform"):
        raise RuntimeError("❌ scaler.pkl did not contain a valid transformer")

    features = joblib.load(features_path)
    if not isinstance(features, (list, tuple)):
        raise ValueError("feature_names.pkl must contain a list of feature names")
    features = list(features)

    return scaler, features

scaler, FEATURES = load_scaler_and_features()

# -----------------------------
# Define categorical features
# -----------------------------
categorical_features = set([
    "HIV+", "def_Anemia", "R_Arth", "c_Pulm", "DM", "htn_C", "hypo_Thy",
    "liver_D", "Mets", "Obesity", "ren_Fail", "Tumor", "MI", "BA", "CVA",
    "ChroLiverDis", "Hemiplegia", "LapCholi", "OpenCholi", "Hernioplasty",
    "Herniotomy", "Lithotomy", "Pyeloplasty", "Appendicectomy", "Omentoplasty",
    "SmallBowelResection", "Laproscopic LysisOfAdhesions", "MRM",
    "Hysterectomy", "Prostectomy", "DiagLaprot", "Nephrectomy", "Gastrectomy",
    "Oesophagotomy", "UnimpDis_LAMA", "SuperficialSSI", "DeepSurgicalSSI",
    "OrganSpaceSSI", "Dehiscence", "GastricOutletObs", "GeneralisedPeritonitis",
    "pul_Complications", "c_Complication", "UTI", "Sepsis", "reoperation", "Readm"
])

# -----------------------------
# Preprocessing Function
# -----------------------------
# ===============================
# Preprocessing Function (Fixed)
# ===============================
# -----------------------------
# Preprocessing Function (Amended)
# -----------------------------
def preprocess_input(df: pd.DataFrame, features: list, scaler):
    """
    Preprocess user input to match model training features:
      - Convert categorical values (Yes/No → 1/0)
      - Add missing features with default 0
      - Reorder features as in training
      - Apply scaling to numeric features
    """
    df_proc = df.copy()

    # Map Yes/No or similar categorical values
    yes_no_map = {"Yes": 1, "No": 0, "Y": 1, "N": 0, "y": 1, "n": 0, 1: 1, 0: 0, "1": 1, "0": 0}

    # Use global categorical_features set
    global categorical_features

    # Encode categorical columns
    for col in categorical_features:
        if col in df_proc.columns:
            df_proc[col] = df_proc[col].map(yes_no_map).fillna(0).astype(int)

    # Add any missing columns with 0
    for feat in features:
        if feat not in df_proc.columns:
            df_proc[feat] = 0

    # Reorder columns exactly as in features list
    df_proc = df_proc[features]

    # Identify numeric features to scale
    numeric_cols = [c for c in features if c not in categorical_features]

    # Apply scaling using array input (avoids feature_names mismatch)
    if numeric_cols:
        try:
            df_proc[numeric_cols] = scaler.transform(df_proc[numeric_cols].values)
        except Exception as e:
            import streamlit as st
            st.error(f"❌ Scaling error: {e}")
            st.stop()

    return df_proc

# -----------------------------
# Google Sheets Integration
# -----------------------------
import gspread
from google.oauth2.service_account import Credentials

# -----------------------------
# Google Sheets Integration for Streamlit App 2
# -----------------------------
def get_gs_client_from_secrets_app2():
    """Authorize Google Sheets client from Streamlit secrets."""
    info = st.secrets.get("gcp_service_account")
    if not info:
        return None, "❌ No gcp_service_account found in Streamlit secrets."
    try:
        creds = Credentials.from_service_account_info(
            info,
            scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
        client = gspread.authorize(creds)
        return client, None
    except Exception as e:
        return None, f"❌ Google Sheets auth failed: {e}"

def ensure_worksheet_exists_app2(sh, worksheet_name):
    """Ensure a worksheet exists in the Google Sheet."""
    try:
        ws = sh.worksheet(worksheet_name)
    except gspread.exceptions.WorksheetNotFound:
        st.warning(f"🧾 Worksheet '{worksheet_name}' not found — creating it now.")
        ws = sh.add_worksheet(title=worksheet_name, rows=100, cols=20)
    return ws

def log_to_gsheet_app2(input_df, preds):
    """Append prediction results to Google Sheet."""
    client, err = get_gs_client_from_secrets_app2()
    if client is None:
        st.warning(f"⚠️ Could not connect to Google Sheets: {err}")
        return

    try:
        sheet_key = st.secrets.get("gsheet_key_streamlit_app_2")
        worksheet_name = st.secrets.get("gsheet_worksheet_app_2", "predictions_app_2")

        sh = client.open_by_key(sheet_key)
        ws = ensure_worksheet_exists_app2(sh, worksheet_name)

        row = input_df.iloc[0].tolist() + [str(preds)]
        ws.append_row(row)
        st.success("✅ Prediction logged to Google Sheet (App 2).")

    except Exception as e:
        st.error(f"❌ Failed to log to Google Sheets: {e}")

def read_from_gsheet_app2(n=5):
    """Read last n rows from the Google Sheet."""
    client, err = get_gs_client_from_secrets_app2()
    if client is None:
        return None, f"❌ Google Sheets client error: {err}"

    try:
        sheet_key = st.secrets.get("gsheet_key_streamlit_app_2")
        worksheet_name = st.secrets.get("gsheet_worksheet_app_2", "predictions_app_2")

        sh = client.open_by_key(sheet_key)
        ws = ensure_worksheet_exists_app2(sh, worksheet_name)

        all_values = ws.get_all_values()
        if not all_values:
            return [], None
        return all_values[-n:], None

    except Exception as e:
        return None, f"❌ Error reading from Google Sheets: {str(e)}"


# -----------------------------
# Input Method
# -----------------------------
input_method = st.radio("Choose input method", ["Manual Entry", "Upload CSV"])
df_input = None

if input_method == "Manual Entry":
    st.info("All features have defaults. Change only the relevant ones.")

    input_dict = {}
    for feat in FEATURES:
        if feat in categorical_features:
            input_dict[feat] = st.selectbox(f"{feat}", ["No", "Yes"], index=0)
        else:
            input_dict[feat] = st.number_input(f"{feat}", step=0.1, value=0.0, format="%.3f")

    if st.button("Predict (Manual)"):
        df_input = pd.DataFrame([input_dict])

elif input_method == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file is not None:
        try:
            df_input = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded file:")
            st.dataframe(df_input.head())
        except Exception as e:
            st.error(f"❌ Could not read uploaded CSV: {e}")
            df_input = None

# -----------------------------
# Prediction Block
# -----------------------------
# -----------------------------
# Prediction Block (Amended)
# -----------------------------
# -----------------------------
# Prediction Block (Amended & Ready)
# -----------------------------
if df_input is not None and not df_input.empty:
    try:
        # 1️⃣ Preprocess input
        df_prepared = preprocess_input(df_input, FEATURES, scaler)
        X_input = df_prepared.values

        # 2️⃣ Shape check
        expected_features_count = len(FEATURES)
        if X_input.shape[1] != expected_features_count:
            st.error(f"❌ Input shape {X_input.shape} does not match expected {expected_features_count} features.")
            st.stop()

        # 3️⃣ Ensemble predictions
        all_probs = ensemble_models_predict_all(X_input, n_forward_passes=10)
        mean_probs = all_probs.mean(axis=0)
        std_devs = all_probs.std(axis=0)
        entropy = calculate_entropy(mean_probs)

        # 4️⃣ Calibration
        calibrated_probs = iso_reg.predict(mean_probs.reshape(-1, 1)).flatten()
        predicted_labels = (calibrated_probs >= best_threshold).astype(int)

        # 5️⃣ Results DataFrame
        results_df = pd.DataFrame({
            "Predicted Label": predicted_labels,
            "Raw Probability": mean_probs,
            "Calibrated Probability": calibrated_probs,
            "Std Dev": std_devs,
            "Entropy": entropy
        })

        st.success("✅ Prediction Completed")
        st.subheader("📊 Prediction Result")
        st.dataframe(results_df)

        # 6️⃣ Log to Google Sheets
        log_to_gsheet(df_input, predicted_labels)

        # 7️⃣ Show latest entries from Google Sheets
        latest_rows, err = read_from_gsheet(n=5)
        if latest_rows is not None and latest_rows:
            st.info("📖 Last 5 rows in Google Sheets:")
            st.dataframe(pd.DataFrame(latest_rows))
        else:
            st.warning(f"⚠️ Could not read back from Google Sheets: {err}")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
