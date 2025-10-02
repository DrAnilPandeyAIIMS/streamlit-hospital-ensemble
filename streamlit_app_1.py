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
        model_probs = [model(input_tensor, training=True).numpy().flatten()
                       for _ in range(n_forward_passes)]
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
    scaler = joblib.load("models/scaler.pkl")
    try:
        features = list(joblib.load("models/feature_names.pkl"))
    except Exception as e:
        st.error(f"❌ Could not load feature_names.pkl: {e}")
        st.stop()
    return scaler, features

scaler, FEATURES = load_scaler_and_features()

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Hospital Predictor - Mobile",
                   layout="centered",
                   initial_sidebar_state="collapsed")

st.title("🏥 Hospital Mortality Predictor (Mobile)")

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

# -----------------------------
# Prediction
# -----------------------------
if input_data is not None and not input_data.empty:
    try:
        X_input = scaler.transform(input_data[FEATURES])
        all_probs = ensemble_models_predict_all(X_input, n_forward_passes=10)

        mean_probs = all_probs.mean(axis=0)
        std_devs = all_probs.std(axis=0)
        entropy = calculate_entropy(mean_probs)

        # ✅ Calibrate
        try:
            calibrated_probs = iso_reg.predict(mean_probs.reshape(-1, 1)).flatten()
        except Exception:
            calibrated_probs = iso_reg.predict(
                np.asarray(mean_probs).reshape(-1, 1)
            ).flatten()

        predicted_labels = (calibrated_probs >= best_threshold).astype(int)

        results_df = pd.DataFrame({
            "Predicted Label": predicted_labels,
            "Raw Probability": mean_probs,
            "Calibrated Probability": calibrated_probs,
            "Std Dev": std_devs,
            "Entropy": entropy
        })

        st.subheader("📊 Prediction Result")
        st.dataframe(results_df)

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
