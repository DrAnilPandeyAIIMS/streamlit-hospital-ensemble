# streamlit_1_app.py
import os
import time
import json
import warnings
import sys
sys.path.append(".")
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import tensorflow_probability as tfp
import joblib
import gdown
from sklearn.isotonic import IsotonicRegression
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import threading

warnings.filterwarnings("ignore")

# -----------------------------
# Model registry
# -----------------------------
MODEL_FILES = {
    "vae_model": {"id": "1GXrJ4GvXOZ4ZzjqQQzfwlyWi9IkSswYe", "path": "models/vae_model.h5"},
    "model_2_probabilistic": {"id": "1ug_BZlcHXwIiOdmC-fnI9SX-ye_ftrad", "path": "models/model_2_probabilistic.h5"},
    "bayesian_model": {"id": "1XIJvwqgakbncaM8QX-BL8ZQ7vMaBWMEp", "path": "models/bayesian_model.h5"},
}

# -----------------------------
# Google Drive download helper
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
            raise FileNotFoundError(f"Download failed, file not found at {path}")
    except Exception as e:
        st.error(f"❌ Could not download model {model_key}: {e}")
        st.stop()

# -----------------------------
# Load model wrapper
# -----------------------------
@st.cache_resource(hash_funcs={dict: lambda _: None})
def load_model_from_drive(model_key):
    path = download_model_if_needed(model_key)
    if os.path.isdir(path) or os.path.isfile(path):
        return tf.keras.models.load_model(path, custom_objects=custom_objects)
    else:
        st.error(f"❌ Model file not found for {model_key}")
        st.stop()

_loaded_models = {}
_loaded_lock = threading.Lock()
def get_model(model_key):
    with _loaded_lock:
        if model_key not in _loaded_models:
            _loaded_models[model_key] = load_model_from_drive(model_key)
    return _loaded_models[model_key]

def preload_all_models():
    for key in MODEL_FILES.keys():
        try:
            get_model(key)
            print(f"✅ Preloaded {key}")
        except Exception as e:
            print(f"⚠️ Could not preload {key}: {e}")

threading.Thread(target=preload_all_models, daemon=True).start()

# -----------------------------
# Custom classes for models
# -----------------------------
class CustomDenseVariational(tfp.layers.DenseVariational):
    def __init__(self, units, make_prior_fn, make_posterior_fn, kl_weight=1.0, **kwargs):
        super().__init__(units=units, make_prior_fn=make_prior_fn, make_posterior_fn=make_posterior_fn, kl_weight=kl_weight, **kwargs)
        self.units = units
        self.kl_weight = kl_weight
    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "kl_weight": self.kl_weight})
        return config
    @classmethod
    def from_config(cls, config):
        if "posterior_fn" not in config:
            config["posterior_fn"] = posterior
        if "prior_fn" not in config:
            config["prior_fn"] = prior
        config["make_prior_fn"] = prior
        config["make_posterior_fn"] = posterior
        return cls(**config)

class DenseFlipoutLayer(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = activation
    def build(self, input_shape):
        self.dense_flipout = tfp.layers.DenseFlipout(units=self.units, activation=self.activation)
        super().build(input_shape)
    def call(self, inputs):
        return self.dense_flipout(inputs)

def build_probabilistic_model(*args, **kwargs):
    raise NotImplementedError("Not available in inference mode.")

def negative_log_likelihood_bernoulli(y_true, y_pred):
    return -tf.reduce_mean(y_true * tf.math.log(y_pred + 1e-9) + (1 - y_true) * tf.math.log(1 - y_pred + 1e-9))

def negative_log_likelihood(y_true, y_pred_dist):
    return -y_pred_dist.log_prob(y_true)

def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential([tfp.layers.VariableLayer(n, dtype=dtype),
                                tfp.layers.DistributionLambda(lambda t: tfp.distributions.MultivariateNormalDiag(loc=t, scale_diag=tf.ones_like(t)))])

def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential([tfp.layers.VariableLayer(tfp.layers.IndependentNormal.params_size(n), dtype=dtype),
                                tfp.layers.IndependentNormal(n, convert_to_tensor_fn=tfp.distributions.Distribution.sample)])

custom_objects = {
    "CustomDenseVariational": CustomDenseVariational,
    "DenseFlipoutLayer": DenseFlipoutLayer,
    "DenseFlipout": tfp.layers.DenseFlipout,
    "DistributionLambda": tfp.layers.DistributionLambda,
    "build_probabilistic_model": build_probabilistic_model,
    "negative_log_likelihood": negative_log_likelihood,
    "negative_log_likelihood_bernoulli": negative_log_likelihood_bernoulli,
    "prior": prior,
    "posterior": posterior,
}

# -----------------------------
# Load ensemble models
# -----------------------------
vae_model = get_model("vae_model")
model_2 = get_model("model_2_probabilistic")
bayesian_model = get_model("bayesian_model")

def ensemble_models_predict_all(input_array, n_forward_passes=10):
    input_tensor = tf.convert_to_tensor(input_array, dtype=tf.float32)
    all_model_probs = []
    for model in [vae_model, model_2, bayesian_model]:
        model_probs = [model(input_tensor, training=True).numpy().flatten() for _ in range(n_forward_passes)]
        all_model_probs.append(np.array(model_probs))
    return np.concatenate(all_model_probs, axis=0)

def calculate_entropy(probs):
    probs = np.clip(np.ravel(probs), 1e-9, 1-1e-9)
    return - (probs * np.log2(probs) + (1 - probs) * np.log2(1 - probs))

def ensure_single_output(prob_vector):
    arr = np.array(prob_vector)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.ravel()
    if (arr < 0).any() or (arr > 1).any():
        arr = 1.0 / (1.0 + np.exp(-arr))
    return arr

# -----------------------------
# Google Sheets helpers
# -----------------------------
def get_gs_client_from_secrets():
    info = st.secrets.get("gcp_service_account")
    if not info:
        return None, "No gcp_service_account found in st.secrets"
    try:
        creds = Credentials.from_service_account_info(info, scopes=["https://www.googleapis.com/auth/spreadsheets"])
        client = gspread.authorize(creds)
        return client, None
    except Exception as e:
        return None, f"Error creating Google Sheets client: {str(e)}"

def _format_cell(v):
    if pd.isna(v): return ""
    if isinstance(v, (pd.Timestamp, datetime)): return v.isoformat()
    if isinstance(v, (np.floating, float)): return float(v)
    if isinstance(v, (np.integer, int)): return int(v)
    return str(v)

def append_to_gsheet(df, sheet_key=None, worksheet_name=None):
    if not isinstance(df, pd.DataFrame):
        try: df = pd.DataFrame(df)
        except Exception as e: return False, f"Could not convert to DataFrame: {e}"
    if df.shape[0] == 0: return True, None
    client, err = get_gs_client_from_secrets()
    if client is None: return False, f"Google Sheets client error: {err}"
    sheet_key = sheet_key or st.secrets.get("gsheet_key")
    worksheet_name = worksheet_name or st.secrets.get("gsheet_worksheet", "predictions")
    if not sheet_key: return False, "No gsheet_key in st.secrets"
    if "docs.google.com" in sheet_key: sheet_key = sheet_key.split("/d/")[1].split("/")[0]
    try:
        sh = client.open_by_key(sheet_key)
        try: ws = sh.worksheet(worksheet_name)
        except gspread.WorksheetNotFound:
            rows = max(1000, df.shape[0]+50)
            cols = max(len(df.columns)+5, 10)
            ws = sh.add_worksheet(title=worksheet_name, rows=str(rows), cols=str(cols))
            ws.append_row(list(df.columns))
        header = ws.row_values(1)
        if not header:
            header = list(df.columns)
            try: ws.insert_row(header, index=1)
            except: ws.append_row(header)
        missing_cols = [c for c in df.columns if c not in header]
        if missing_cols:
            new_header = header + missing_cols
            try: ws.update("A1", [new_header])
            except: pass
        rows = []
        for _, r in df.iterrows():
            row_values = [_format_cell(r[col]) if col in df.columns else "" for col in header]
            rows.append(row_values)
        try: ws.append_rows(rows, value_input_option="USER_ENTERED")
        except: 
            for row_values in rows: ws.append_row(row_values)
        return True, None
    except Exception as e:
        return False, str(e)

def read_from_gsheet(n=5):
    client, err = get_gs_client_from_secrets()
    if client is None: return None, f"Google Sheets client error: {err}"
    sheet_key = st.secrets.get("gsheet_key")
    worksheet_name = st.secrets.get("gsheet_worksheet", "predictions")
    if not sheet_key: return None, "No gsheet_key in st.secrets"
    try:
        sh = client.open_by_key(sheet_key)
        ws = sh.worksheet(worksheet_name)
        all_values = ws.get_all_values()
        if not all_values: return [], None
        df = pd.DataFrame(all_values[1:], columns=all_values[0])
        return df.tail(n), None
    except Exception as e:
        return None, str(e)

# -----------------------------
# Streamlit config
# -----------------------------
st.set_page_config(page_title="Hospital Predictor - Mobile", layout="centered", initial_sidebar_state="collapsed")
st.title("🏥 Hospital Mortality Predictor (Mobile)")
st.markdown("Upload patient CSV or enter manually to get predictions.")

# -----------------------------
# Load scaler and features
# -----------------------------
@st.cache_resource
def load_scaler_and_features():
    obj = joblib.load("models/scaler.pkl")
    features = obj.get("features")
    scaler = obj.get("scaler")
    return scaler, features

scaler, FEATURES = load_scaler_and_features()

# -----------------------------
# Input section
# -----------------------------
input_method = st.radio("Choose input method", ["Manual Entry", "Upload CSV"])

input_data = None
if input_method == "Manual Entry":
    input_dict = {}
    for f in FEATURES:
        input_dict[f] = st.text_input(f, "")
    if st.button("Predict"):
        input_data = pd.DataFrame([input_dict])
elif input_method == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file:
        input_data = pd.read_csv(uploaded_file)
        if st.button("Predict"):
            pass

if input_data is not None and not input_data.empty:
    try:
        X_input = scaler.transform(input_data[FEATURES])
        all_probs = ensemble_models_predict_all(X_input, n_forward_passes=10)
        mean_probs = all_probs.mean(axis=0)
        std_devs = all_probs.std(axis=0)
        entropy = calculate_entropy(mean_probs)
        predicted_labels = (mean_probs >= 0.6).astype(int)
        results_df = pd.DataFrame({
            "Predicted Label": predicted_labels,
            "Mean Prob": mean_probs,
            "Std Dev": std_devs,
            "Entropy": entropy
        })
        st.subheader("📊 Prediction Result")
        st.write(f"**Predicted Label:** {predicted_labels[0]}")
        st.write(f"**Mean Probability:** {mean_probs[0]:.3f}")
        st.write(f"**Std Dev / Uncertainty:** {std_devs[0]:.3f}")
        st.write(f"**Entropy:** {entropy[0]:.3f}")
        success, msg = append_to_gsheet(results_df)
        if success:
            st.success("✅ Prediction saved to Google Sheets.")
        else:
            st.warning(f"⚠️ Could not save to Google Sheets: {msg}")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
