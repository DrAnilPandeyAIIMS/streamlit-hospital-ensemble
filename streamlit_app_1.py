# streamlit_app_1.py
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
    "model_2_probabilistic": {
        "id": "1ug_BZlcHXwIiOdmC-fnI9SX-ye_ftrad",
        "path": "models/model_2_probabilistic.h5",
    },
    "bayesian_model": {
        "id": "1XIJvwqgakbncaM8QX-BL8ZQ7vMaBWMEp",
        "path": "models/bayesian_model.h5",
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
class CustomDenseVariational(tfp.layers.DenseVariational):
    def __init__(self, units, make_prior_fn, make_posterior_fn, kl_weight=1.0, **kwargs):
        super().__init__(units=units,
                         make_prior_fn=make_prior_fn,
                         make_posterior_fn=make_posterior_fn,
                         kl_weight=kl_weight,
                         **kwargs)
        self.units = units
        self.kl_weight = kl_weight
    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "kl_weight": self.kl_weight})
        return config

class DenseFlipoutLayer(tf.keras.layers.Layer):
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

def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(n, dtype=dtype),
        tfp.layers.DistributionLambda(
            lambda t: tfp.distributions.MultivariateNormalDiag(
                loc=t, scale_diag=tf.ones_like(t)
            )
        )
    ])

def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(
            tfp.layers.IndependentNormal.params_size(n), dtype=dtype
        ),
        tfp.layers.IndependentNormal(n,
            convert_to_tensor_fn=tfp.distributions.Distribution.sample)
    ])

def negative_log_likelihood(y_true, y_pred_dist):
    return -y_pred_dist.log_prob(y_true)

custom_objects = {
    "CustomDenseVariational": CustomDenseVariational,
    "DenseFlipoutLayer": DenseFlipoutLayer,
    "DenseFlipout": tfp.layers.DenseFlipout,
    "DistributionLambda": tfp.layers.DistributionLambda,
    "prior": prior,
    "posterior": posterior,
    "negative_log_likelihood": negative_log_likelihood,
}

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
model_2 = get_model("model_2_probabilistic")
bayesian_model = get_model("bayesian_model")

# -----------------------------
# Ensemble predictions
# -----------------------------
def ensemble_models_predict_all(input_array, n_forward_passes=10):
    input_tensor = tf.convert_to_tensor(input_array, dtype=tf.float32)   
    all_model_probs = []
    for model in [vae_model, model_2, bayesian_model]:
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
def preprocess_input(df: pd.DataFrame, features: list, scaler, categorical_features: list):
    df_proc = df.copy()
    yes_no_map = {"Yes":1,"No":0,"Y":1,"N":0,"y":1,"n":0,1:1,0:0,"1":1,"0":0}

    # Map categorical features
    for col in categorical_features:
        if col in df_proc.columns:
            df_proc[col] = df_proc[col].map(yes_no_map).fillna(0).astype(int)

    # Always use the same feature ordering from feature_names.pkl
    expected_features = features

    # Align input dataframe with expected feature list
    df_proc = df_proc.reindex(columns=expected_features, fill_value=0)

    # Scale only numeric features (non-categorical)
    numeric_to_scale = [c for c in expected_features if c not in categorical_features]
    if numeric_to_scale:
        try:
            df_proc[numeric_to_scale] = scaler.transform(df_proc[numeric_to_scale])
        except Exception as e:
            st.error(f"❌ Scaling error: {e}")
            st.stop()

    return df_proc



# -----------------------------
# Google Sheets Integration
# -----------------------------
def get_gs_client_from_secrets():
    info = st.secrets.get("gcp_service_account")
    if not info:
        return None, "No gcp_service_account found in st.secrets"
    try:
        creds = Credentials.from_service_account_info(
            info,
            scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
        client = gspread.authorize(creds)
        return client, None
    except Exception as e:
        return None, f"Error creating Google Sheets client: {str(e)}"

def log_to_gsheet(input_df, preds):
    client, err = get_gs_client_from_secrets()
    if client is None:
        st.warning(f"⚠️ Could not save to Google Sheets: {err}")
        return
    try:
        sheet_key = st.secrets.get("gsheet_key")
        worksheet_name = st.secrets.get("gsheet_worksheet", "predictions")
        sh = client.open_by_key(sheet_key)
        ws = sh.worksheet(worksheet_name)
        row = input_df.iloc[0].tolist() + [str(preds)]
        ws.append_row(row)
        st.success("✅ Prediction logged to Google Sheets")
    except Exception as e:
        st.error(f"❌ Failed to log to Google Sheet: {e}")

def read_from_gsheet(n=5):
    client, err = get_gs_client_from_secrets()
    if client is None:
        return None, f"Google Sheets client error: {err}"
    try:
        sheet_key = st.secrets.get("gsheet_key")
        worksheet_name = st.secrets.get("gsheet_worksheet", "predictions")
        sh = client.open_by_key(sheet_key)
        ws = sh.worksheet(worksheet_name)
        all_values = ws.get_all_values()
        if not all_values:
            return [], None
        return all_values[-n:], None
    except Exception as e:
        return None, f"Error reading from Google Sheets: {str(e)}"

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
if df_input is not None and not df_input.empty:
    try:
        df_prepared = preprocess_input(df_input, FEATURES, scaler)
        X_input = df_prepared.values

        all_probs = ensemble_models_predict_all(X_input, n_forward_passes=10)
        mean_probs = all_probs.mean(axis=0)
        std_devs = all_probs.std(axis=0)
        entropy = calculate_entropy(mean_probs)

        calibrated_probs = iso_reg.predict(mean_probs.reshape(-1, 1)).flatten()
        predicted_labels = (calibrated_probs >= best_threshold).astype(int)

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

        log_to_gsheet(df_input, predicted_labels)

        latest_rows, err = read_from_gsheet(n=5)
        if latest_rows is not None and latest_rows:
            st.info("📖 Last 5 rows in Google Sheets:")
            st.dataframe(pd.DataFrame(latest_rows))
        else:
            st.warning(f"⚠️ Could not read back from Google Sheets: {err}")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
