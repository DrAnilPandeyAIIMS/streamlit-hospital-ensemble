# streamlit_app.py

# -----------------------------
# Imports
# -----------------------------
import os
import time
import json
import warnings
import sys
sys.path.append(".")  # <-- ensure local modules like custom_layers are found
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import tensorflow_probability as tfp
import joblib
import gdown
from sklearn.isotonic import IsotonicRegression
from custom_layers import (
    DenseFlipoutLayer,
    negative_log_likelihood_bernoulli,
    build_probabilistic_model,
)

warnings.filterwarnings("ignore")

# -----------------------------
# Streamlit page configuration
# -----------------------------
st.set_page_config(
    page_title="Hospital Ensemble Predictor",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Helpful header info
st.write("Current working directory:", os.getcwd())
st.title("🏥 Hospital Mortality Predictor")
st.markdown("Upload patient data or enter values to get predictions.")

# -----------------------------
# Clear cache button
# -----------------------------
with st.sidebar:
    if st.button("🔄 Clear Cache & Rerun"):
        try:
            st.cache_data.clear()
        except Exception:
            pass
        try:
            st.cache_resource.clear()
        except Exception:
            pass
        st.success("✅ Cache cleared. Rerunning...")
        time.sleep(0.6)
        st.rerun()

# -----------------------------
# Load scalers, features, and isotonic regressor
# -----------------------------
# -----------------------------
# Load scalers, features, and isotonic regressor
# -----------------------------
try:
    scaler = joblib.load("models/scaler.pkl")

    # Handle case where scaler.pkl accidentally contains a tuple
    if isinstance(scaler, tuple):
        st.warning("⚠️ scaler.pkl contained a tuple. Using the first element as scaler.")
        scaler = scaler[0]

    st.write("Scaler type:", type(scaler))  # debug info, safe to remove later

except Exception as e:
    st.error(f"❌ Could not load scaler.pkl from models/. Error: {e}")
    st.stop()

try:
    feature_names = joblib.load("models/feature_names.pkl")
    # Ensure feature_names is a list for indexing order
    feature_names = list(feature_names)
except Exception as e:
    st.error(f"❌ Could not load feature_names.pkl from models/. Error: {e}")
    st.stop()

try:
    iso_reg = joblib.load("models/iso_reg.pkl")
except Exception as e:
    st.error(f"❌ Could not load iso_reg.pkl from models/. Error: {e}")
    st.stop()

try:
    with open("best_threshold.json", "r") as f:
        best_threshold = json.load(f)["best_threshold"]
except Exception as e:
    st.error(f"❌ Could not load best_threshold.json. Error: {e}")
    st.stop()

# -----------------------------
# TensorFlow Probability setup
# -----------------------------
tfd = tfp.distributions

def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(n, dtype=dtype),
        tfp.layers.DistributionLambda(
            lambda t: tfd.MultivariateNormalDiag(
                loc=t,
                scale_diag=tf.ones_like(t)
            )
        )
    ])

def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(tfp.layers.IndependentNormal.params_size(n), dtype=dtype),
        tfp.layers.IndependentNormal(n, convert_to_tensor_fn=tfd.Distribution.sample),
    ])

# -----------------------------
# Custom Dense Variational Layer
# -----------------------------
class CustomDenseVariational(tfp.layers.DenseVariational):
    def __init__(self, units, make_prior_fn, make_posterior_fn, kl_weight=1.0, **kwargs):
        super().__init__(
            units=units,
            make_prior_fn=make_prior_fn,
            make_posterior_fn=make_posterior_fn,
            kl_weight=kl_weight,
            **kwargs,
        )
        self.units = units
        self.make_prior_fn = make_prior_fn
        self.make_posterior_fn = make_posterior_fn
        self.kl_weight = kl_weight

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "make_prior_fn": "prior",
            "make_posterior_fn": "posterior",
            "kl_weight": self.kl_weight,
        })
        return config

    @classmethod
    def from_config(cls, config):
        config["make_prior_fn"] = prior
        config["make_posterior_fn"] = posterior
        return cls(**config)

custom_objects = {
    "CustomDenseVariational": CustomDenseVariational,
    "negative_log_likelihood": negative_log_likelihood_bernoulli,
    "negative_log_likelihood_bernoulli": negative_log_likelihood_bernoulli,
    "prior": prior,
    "posterior": posterior,
    "DenseFlipoutLayer": DenseFlipoutLayer,
    "DenseFlipout": tfp.layers.DenseFlipout,
}

# -----------------------------
# Google Drive file mapping
# -----------------------------
# NOTE: Corrected Bayesian model ID (observed earlier typo with 'Q' vs 'Z')
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
        "id": "1XIJvwqgakbncaM8QX-BL8ZQ7vMaBWMEp",  # ← ensure this is correct
        "path": "models/bayesian_model.h5",
    },
}

def _ensure_models_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def download_from_gdrive(file_id, output_path):
    """Download a file from Google Drive if not already present locally"""
    _ensure_models_dir(output_path)
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        with st.spinner(f"📥 Downloading {os.path.basename(output_path)} from Google Drive..."):
            try:
                gdown.download(url, output_path, quiet=False)
            except Exception as e:
                st.error(f"❌ Download failed for {output_path}. Error: {e}")
                st.stop()

def load_model_from_drive(model_key, custom_objects=None):
    """Ensure model file is available, then load with tf.keras"""
    info = MODEL_FILES[model_key]
    download_from_gdrive(info["id"], info["path"])
    try:
        with st.spinner(f"🔧 Loading {model_key}..."):
            model = tf.keras.models.load_model(info["path"], custom_objects=custom_objects)
        return model
    except Exception as e:
        st.error(f"❌ Failed to load {model_key} from {info['path']}. Error: {e}")
        st.stop()

# -----------------------------
# Load models safely (Drive-aware)
# -----------------------------
vae_model = load_model_from_drive("vae_model")
model_2 = load_model_from_drive("model_2_probabilistic", custom_objects=custom_objects)
bayesian_model = load_model_from_drive("bayesian_model", custom_objects=custom_objects)
ensemble_models = [vae_model, model_2, bayesian_model]

# -----------------------------
# Ensemble prediction function
# -----------------------------
def ensemble_models_predict_all(input_array, n_forward_passes=100):
    """
    Runs:
      - Deterministic pass on VAE head (replicated n times)
      - n stochastic passes on Flipout model and Bayesian model
    Returns a (n_models_passes, n_samples) array of probabilities.
    """
    input_tensor = tf.convert_to_tensor(input_array, dtype=tf.float32)
    all_model_probs = []

    # VAE: deterministic forward, replicate to match sampling size
    vae_probs = vae_model(input_tensor, training=False).numpy().flatten()
    vae_stack = np.stack([vae_probs] * n_forward_passes)
    all_model_probs.append(vae_stack)

    # Stochastic models: sample with training=True
    for model in [model_2, bayesian_model]:
        model_probs = [
            model(input_tensor, training=True).numpy().flatten()
            for _ in range(n_forward_passes)
        ]
        all_model_probs.append(np.array(model_probs))

    all_model_probs = np.concatenate(all_model_probs, axis=0)  # shape: (3 * n_forward, N)
    return all_model_probs

# -----------------------------
# Helpers
# -----------------------------
def calculate_entropy(probs):
    probs = np.clip(np.ravel(probs), 1e-9, 1 - 1e-9)  # ensure 1D
    return - (probs * np.log2(probs) + (1 - probs) * np.log2(1 - probs))

def ensure_single_output(prob_vector):
    """
    Ensure the model output is a single column per sample.
    If model returns Nx1, squeeze; if it returns logits, apply sigmoid.
    """
    arr = np.array(prob_vector)
    # If shape is (N,1) → flatten
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.ravel()
    # If any values are outside [0,1], apply sigmoid as a fallback
    if (arr < 0).any() or (arr > 1).any():
        arr = 1.0 / (1.0 + np.exp(-arr))
    return arr

# -----------------------------
# Streamlit UI – File upload path
# -----------------------------
st.header("Ensemble Model Predictor")
uploaded_file = st.file_uploader("Upload Patient CSV File", type=["csv"])

if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"❌ Could not read CSV: {e}")
        st.stop()

    st.write("Uploaded Data:")
    st.dataframe(input_df)

    missing_cols = set(feature_names) - set(input_df.columns)
    if missing_cols:
        st.error(f"❌ Missing required fields: {missing_cols}")
    else:
        # Reorder & fill
        input_df = input_df[feature_names].fillna(0)

        # Scale
        try:
            input_df[input_df.columns] = scaler.transform(input_df[input_df.columns])
        except Exception as e:
            st.error(f"❌ Scaling error: {e}")
            st.stop()

        input_array = input_df.values

        # Predict ensemble
        all_probs = ensemble_models_predict_all(input_array)
        mean_probs = np.mean(all_probs, axis=0)
        mean_probs = ensure_single_output(mean_probs)  # Ensure single column probabilities

        std_devs = np.std(all_probs, axis=0)
        entropy = calculate_entropy(mean_probs)

        # Isotonic calibration expects 1D (same type as trained)
        try:
            calibrated_probs = iso_reg.predict(mean_probs)
        except Exception:
            # Some versions expect shape (n_samples,)
            calibrated_probs = iso_reg.predict(np.asarray(mean_probs))

        predicted_labels = (calibrated_probs >= best_threshold).astype(int)

        # Results
        results_df = input_df.copy()
        results_df["raw_probability"] = mean_probs
        results_df["calibrated_probability"] = calibrated_probs
        results_df["predicted_label"] = predicted_labels
        results_df["std_deviation"] = std_devs
        results_df["entropy"] = entropy

        st.subheader("Prediction Results")
        st.dataframe(results_df)

        st.download_button(
            "Download Results as CSV",
            results_df.to_csv(index=False),
            "predictions.csv",
            "text/csv",
        )

# -----------------------------
# Streamlit UI – Manual entry path
# -----------------------------
else:
    st.subheader("📋 Enter Patient Data Manually")
    with st.form("manual_form"):
        manual_data = {}
        for feature in feature_names:
            # Keep your original boolean-style mapping if you had any; otherwise numeric
            if feature.lower() in ["comorbidity", "on_ventilator", "diabetic", "hypertensive"]:
                manual_data[feature] = st.selectbox(f"{feature}", ["No", "Yes"], index=0)
            else:
                manual_data[feature] = st.number_input(f"{feature}", step=0.1, value=0.0)

        submitted = st.form_submit_button("Predict")

        if submitted:
            df_input = pd.DataFrame([manual_data])

            # Convert Yes/No → 1/0
            for col in df_input.columns:
                if df_input[col].dtype == object:
                    df_input[col] = df_input[col].map({"Yes": 1, "No": 0}).fillna(0)

            df_input = df_input[feature_names].fillna(0)

            try:
                df_input[df_input.columns] = scaler.transform(df_input[df_input.columns])
            except Exception as e:
                st.error(f"❌ Scaling error: {e}")
                st.stop()

            input_array = df_input.values

            # Predict ensemble
            all_probs = ensemble_models_predict_all(input_array)
            mean_probs = np.mean(all_probs, axis=0)
            mean_probs = ensure_single_output(mean_probs)

            std_devs = np.std(all_probs, axis=0)
            entropy = calculate_entropy(mean_probs)

            # Isotonic calibration
            try:
                calibrated_probs = iso_reg.predict(mean_probs)
            except Exception:
                calibrated_probs = iso_reg.predict(np.asarray(mean_probs))

            predicted_labels = (calibrated_probs >= best_threshold).astype(int)

            st.subheader("Prediction Result")
            st.write(f"**Raw Probability:** {mean_probs[0]:.3f}")
            st.write(f"**Calibrated Probability:** {calibrated_probs[0]:.3f}")
            st.write(f"**Predicted Label:** {'High Risk' if predicted_labels[0] == 1 else 'Low Risk'}")
            st.write(f"**Uncertainty (Std Dev):** {std_devs[0]:.3f}")
            st.write(f"**Entropy:** {entropy[0]:.3f}")
