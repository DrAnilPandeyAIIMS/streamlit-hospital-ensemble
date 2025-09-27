# streamlit_app.py`
import os
import time
import json
import warnings
import sys
import warnings
sys.path.append(".")
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import tensorflow_probability as tfp
import joblib
import gdown
from sklearn.isotonic import IsotonicRegression

# Google Sheets
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import streamlit as st
from datetime import datetime
import threading
# --------------------------
# Google Sheets Connection
# --------------------------
# -----------------------------
# Model metadata (Google Drive IDs + local paths)
# -----------------------------
MODEL_FILES = { 
    "vae_model": {
        "id": "1GXrJ4GvXOZ4ZzjqQQzfwlyWi9IkSswYe",
        "path": "models/vae_model.h5"
    },
    "model_2_probabilistic": {
        "id": "1ug_BZlcHXwIiOdmC-fnI9SX-ye_ftrad",
        "path": "models/model_2_probabilistic.h5"
    },
    "bayesian_model": {
        "id": "1XIJvwqgakbncaM8QX-BL8ZQ7vMaBWMEp",
        "path": "models/bayesian_model.h5"
    },
}    


# -----------------------------
# Ensure local directory exists
# -----------------------------
def _ensure_models_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# -----------------------------
# Download from Google Drive if not present
# -----------------------------
def download_model_if_needed(model_key):
    info = MODEL_FILES[model_key]
    _ensure_models_dir(info["path"])
    if not os.path.exists(info["path"]) and not os.path.exists(info["path"] + ".h5"):
        url = f"https://drive.google.com/uc?id={info['id']}"
        with st.spinner(f"📥 Downloading {os.path.basename(info['path'])}..."):
            gdown.download(url, info["path"], quiet=False)
    return info["path"]

# -----------------------------
# Cache-friendly loader
# -----------------------------
@st.cache_resource(hash_funcs={dict: lambda _: None})
def load_model_from_drive(model_key):
    path = download_model_if_needed(model_key)

    # SavedModel directory
    if os.path.isdir(path):
        return tf.keras.models.load_model(path, custom_objects=custom_objects)
    # HDF5 fallback
    elif os.path.isfile(path + ".h5"):
        return tf.keras.models.load_model(path + ".h5", custom_objects=custom_objects)
    else:
        st.error(f"❌ Model file not found for {model_key}")
        st.stop()

# -----------------------------
# Thread-safe preload dictionary
# -----------------------------
_loaded_models = {}
_loaded_lock = threading.Lock()

def get_model(model_key):
    """Safe access to models. Guarantees the model is loaded."""
    with _loaded_lock:
        if model_key not in _loaded_models:
            _loaded_models[model_key] = load_model_from_drive(model_key)
    return _loaded_models[model_key]

# -----------------------------
# Background preloading (non-blocking)
# -----------------------------
def preload_all_models():
    for key in MODEL_FILES.keys():
        try:
            get_model(key)
            print(f"✅ Preloaded {key}")
        except Exception as e:
            print(f"⚠️ Could not preload {key}: {e}")

threading.Thread(target=preload_all_models, daemon=True).start()

# -----------------------------
# Ensemble safe loader
# -----------------------------
# =================================================
# Google Sheets helpers (kept once, not duplicated)
# =================================================
def _format_cell(v):
    if pd.isna(v):
        return ""
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.isoformat()
    if isinstance(v, (np.floating, float)):
        return float(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    return str(v)

def append_to_gsheet(df, sheet_key=None, worksheet_name=None):
    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except Exception as e:
            return False, f"append_to_gsheet: could not convert input to DataFrame: {e}"

    if df.shape[0] == 0:
        return True, None

    client, err = get_gs_client_from_secrets()
    if client is None:
        return False, f"Google Sheets client error: {err}"

    sheet_key = sheet_key or st.secrets.get("gsheet_key")
    worksheet_name = worksheet_name or st.secrets.get("gsheet_worksheet", "predictions")

    if not sheet_key:
        return False, "No gsheet_key in st.secrets"

    if "docs.google.com" in sheet_key:
        try:
            sheet_key = sheet_key.split("/d/")[1].split("/")[0]
        except Exception:
            return False, "Invalid gsheet_key format. Must be the spreadsheet ID."

    try:
        sh = client.open_by_key(sheet_key)

        try:
            ws = sh.worksheet(worksheet_name)
        except gspread.WorksheetNotFound:
            rows = max(1000, df.shape[0] + 50)
            cols = max(len(df.columns) + 5, 10)
            ws = sh.add_worksheet(title=worksheet_name, rows=str(rows), cols=str(cols))
            ws.append_row(list(df.columns))

        header = ws.row_values(1)
        if not header:
            header = list(df.columns)
            try:
                ws.insert_row(header, index=1)
            except Exception:
                ws.append_row(header)

        missing_cols = [c for c in list(df.columns) if c not in header]
        if missing_cols:
            new_header = header + missing_cols
            try:
                ws.update("A1", [new_header])
                header = new_header
            except Exception:
                try:
                    ws.delete_rows(1)
                except Exception:
                    pass
                try:
                    ws.insert_row(new_header, index=1)
                    header = new_header
                except Exception:
                    header = header

        rows = []
        for _, r in df.iterrows():
            row_values = []
            for col in header:
                if col in df.columns:
                    val = r[col]
                    row_values.append(_format_cell(val))
                else:
                    row_values.append("")
            rows.append(row_values)

        try:
            ws.append_rows(rows, value_input_option="USER_ENTERED")
        except Exception:
            for row_values in rows:
                ws.append_row(row_values)
        return True, None

    except Exception as e:
        return False, str(e)

# --------------------------
# Write Predictions
# --------------------------
def write_prediction(raw_prob, calibrated_prob, predicted_label, uncertainty):
    client, err = get_gs_client_from_secrets()
    if client is None:
        return False, f"Google Sheets client error: {err}"

    sheet_key = st.secrets.get("gsheet_key")
    worksheet_name = st.secrets.get("gsheet_worksheet", "predictions")

    if not sheet_key:
        return False, "No gsheet_key in st.secrets"

    try:
        sh = client.open_by_key(sheet_key)
        try:
            ws = sh.worksheet(worksheet_name)
        except gspread.WorksheetNotFound:
            ws = sh.add_worksheet(title=worksheet_name, rows="1000", cols="10")
            ws.append_row(["Timestamp", "Raw Probability", "Calibrated Probability", "Predicted Label", "Uncertainty"])

        # Append a single row
        ws.append_row([
            str(datetime.now()),
            round(raw_prob, 3),
            round(calibrated_prob, 3),
            predicted_label,
            round(uncertainty, 3)
        ])

        return True, None
    except Exception as e:
        return False, str(e)


# --------------------------
# Read Last N Predictions
# --------------------------
def read_from_gsheet(n=5):
    client, err = get_gs_client_from_secrets()
    if client is None:
        return None, f"Google Sheets client error: {err}"

    sheet_key = st.secrets.get("gsheet_key")
    worksheet_name = st.secrets.get("gsheet_worksheet", "predictions")

    if not sheet_key:
        return None, "No gsheet_key in st.secrets"

    try:
        sh = client.open_by_key(sheet_key)
        ws = sh.worksheet(worksheet_name)

        all_values = ws.get_all_values()
        if not all_values:
            return [], None

        # Convert to dataframe for nicer display
        df = pd.DataFrame(all_values[1:], columns=all_values[0])  # first row = header
        latest = df.tail(n)
        return latest, None
    except Exception as e:
        return None, str(e)


warnings.filterwarnings("ignore")
# -----------------------------
# Streamlit page configuration
# -----------------------------
st.set_page_config(
    page_title="Hospital Ensemble Predictor",
    layout="centered",
    initial_sidebar_state="collapsed",
)

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
# Helpers: load scaler robustly
# -----------------------------
@st.cache_resource
def load_scaler_and_features():
    """
    Load models/scaler.pkl which may be:
      - a scaler object
      - (scaler, list_of_columns_to_scale)
      - nested tuple variations
    Returns (scaler, cols_list_or_None)
    """
    obj = joblib.load("models/scaler.pkl")

    # unwrap defensively
    scaler = None
    cols = None

    if isinstance(obj, tuple):
        # common case: (scaler, cols)
        if hasattr(obj[0], "transform"):
            scaler = obj[0]
            cols = obj[1] if len(obj) > 1 else None
        elif isinstance(obj[0], tuple) and hasattr(obj[0][0], "transform"):
            scaler = obj[0][0]
            cols = obj[0][1] if len(obj[0]) > 1 else None
        else:
            # fallback: try to find first transformable element
            for el in obj:
                if hasattr(el, "transform"):
                    scaler = el
                    break
    else:
        if hasattr(obj, "transform"):
            scaler = obj
        else:
            scaler = None

    if scaler is None or not hasattr(scaler, "transform"):
        raise RuntimeError(f"Loaded scaler.pkl did not produce a transformer. Got: {type(obj)}")

    return scaler, cols

scaler, scaler_features = load_scaler_and_features()

# -----------------------------
# Load feature names & iso reg / threshold
# -----------------------------
try:
    feature_names = list(joblib.load("models/feature_names.pkl"))
except Exception as e:
    st.error(f"❌ Could not load feature_names.pkl. Error: {e}")
    st.stop()

@st.cache_resource
def load_iso_reg():
    return joblib.load("models/iso_reg.pkl")

iso_reg = load_iso_reg()

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
# Custom classes (for loading models)
# -----------------------------
# Minimal stub to allow loading
class CustomDenseVariational(tfp.layers.DenseVariational):
    def __init__(self, units, make_prior_fn, make_posterior_fn, kl_weight=1.0, **kwargs):
        super().__init__(
            units=units,
            make_prior_fn=make_prior_fn,
            make_posterior_fn=make_posterior_fn,
            kl_weight=kl_weight,
            **kwargs
        )
        self.units = units
        self.kl_weight = kl_weight

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "kl_weight": self.kl_weight
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Ensure both old keys from saved model and new keys are set
        if "posterior_fn" not in config:
            config["posterior_fn"] = posterior
        if "prior_fn" not in config:
            config["prior_fn"] = prior
        config["make_prior_fn"] = prior
        config["make_posterior_fn"] = posterior
        return cls(**config)


# Negative log likelihood for Bernoulli outputs

# ✅ Bernoulli NLL (binary classification probabilistic output)
# Placeholder for build_probabilistic_model since it's only needed during training
def build_probabilistic_model(*args, **kwargs):
    raise NotImplementedError("build_probabilistic_model is not available in inference mode.")
def negative_log_likelihood_bernoulli(y_true, y_pred):
    return -tf.reduce_mean(
        y_true * tf.math.log(y_pred + 1e-9) +
        (1 - y_true) * tf.math.log(1 - y_pred + 1e-9)
    )

# ✅ Generic NLL for distributions (used with tfp.layers.DistributionLambda)
def negative_log_likelihood(y_true, y_pred_dist):
    """
    y_pred_dist is a Distribution object (e.g., Bernoulli, Normal, etc.)
    This computes the mean negative log-likelihood.
    """
    return -y_pred_dist.log_prob(y_true)

# Your probabilistic layers
# Your probabilistic DenseFlipout wrapper
class DenseFlipoutLayer(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        self.dense_flipout = tfp.layers.DenseFlipout(
            units=self.units, 
            activation=self.activation
        )
        super().build(input_shape)

    def call(self, inputs):
        return self.dense_flipout(inputs)


# ✅ Register all custom objects so model loading works
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
# Load models once (ensemble components)
# -----------------------------
vae_model = get_model("vae_model")
model_2 = get_model("model_2_probabilistic")
bayesian_model = get_model("bayesian_model")


# -----------------------------
# Ensemble prediction function
# -----------------------------
def ensemble_models_predict_all(input_array, n_forward_passes=100):
    input_tensor = tf.convert_to_tensor(input_array, dtype=tf.float32)
    all_model_probs = []

    vae_probs = vae_model(input_tensor, training=False).numpy().flatten()
    vae_stack = np.stack([vae_probs] * n_forward_passes)
    all_model_probs.append(vae_stack)

    for model in [model_2, bayesian_model]:
        model_probs = [
            model(input_tensor, training=True).numpy().flatten()
            for _ in range(n_forward_passes)
        ]
        all_model_probs.append(np.array(model_probs))

    all_model_probs = np.concatenate(all_model_probs, axis=0)
    return all_model_probs

# -----------------------------
# Misc helpers
# -----------------------------
def calculate_entropy(probs):
    probs = np.clip(np.ravel(probs), 1e-9, 1 - 1e-9)
    return - (probs * np.log2(probs) + (1 - probs) * np.log2(1 - probs))

def ensure_single_output(prob_vector):
    arr = np.array(prob_vector)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.ravel()
    if (arr < 0).any() or (arr > 1).any():
        arr = 1.0 / (1.0 + np.exp(-arr))
    return arr

def scale_dataframe(df: pd.DataFrame, scaler, cols_to_scale=None) -> pd.DataFrame:
    """
    Fill NaNs, then scale either:
      - the provided cols_to_scale (if not None)
      - otherwise, try to scale numeric columns only
    """
    df = df.copy().fillna(0.0)

    if cols_to_scale is not None and len(cols_to_scale) > 0:
        cols = [c for c in cols_to_scale if c in df.columns]
        if len(cols) == 0:
            # nothing to scale
            return df
        # ensure numeric dtype
        df[cols] = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        df[cols] = scaler.transform(df[cols])
    else:
        # infer numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 0:
            df[numeric_cols] = scaler.transform(df[numeric_cols])
    return df

# -----------------------------
# Google Sheets helper
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

    
# -----------------------------
# Ensure local directory exists
# -----------------------------

# -----------------------------
# Download from Google Drive if not present
# -----------------------------


# -----------------------------
# Cache-friendly loader
# -----------------------------

# -----------------------------
# UI — File upload path (single-column outcome)
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

    # Warn about extra columns
    extra_cols = [c for c in input_df.columns if c not in feature_names]
    if extra_cols:
        st.warning(f"⚠️ Extra columns ignored: {extra_cols}")

    # Add missing columns and reorder
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0.0
    input_df = input_df[feature_names].fillna(0.0)

    # Scale numeric features
    try:
        df_scaled = input_df.copy()
        numeric_to_scale = [
            col for col in scaler_features 
            if col in df_scaled.columns and col not in categorical_features
        ]
        if numeric_to_scale:
            df_scaled[numeric_to_scale] = scaler.transform(df_scaled[numeric_to_scale])
        input_df = df_scaled
    except Exception as e:
        st.error(f"❌ Scaling error: {e}")
        st.stop()

    # Convert to array
    input_array = input_df.values

    # Predict ensemble
    all_probs = ensemble_models_predict_all(input_array)
    mean_probs = np.mean(all_probs, axis=0)      # 1D
    mean_probs = ensure_single_output(mean_probs)
    std_devs = np.std(all_probs, axis=0)         # 1D
    entropy = calculate_entropy(mean_probs)      # 1D

    # Calibrate probabilities
    try:
        calibrated_probs = iso_reg.predict(mean_probs.reshape(-1, 1)).flatten()
    except Exception:
        calibrated_probs = iso_reg.predict(np.asarray(mean_probs).reshape(-1, 1)).flatten()

    predicted_labels = (calibrated_probs >= best_threshold).astype(int)

    # Construct results DataFrame
    results_df = input_df.copy()
    results_df["raw_probability"] = mean_probs
    results_df["calibrated_probability"] = calibrated_probs
    results_df["predicted_label"] = predicted_labels
    results_df["std_deviation"] = std_devs
    results_df["entropy"] = entropy

    st.subheader("Prediction Results")
    st.dataframe(results_df)

    # Save to Google Sheets
    ok, err = append_to_gsheet(results_df)
    if ok:
        st.success("✅ Saved prediction to Google Sheets.")

        latest_rows, err2 = read_from_gsheet(n=5)
        if latest_rows:
            st.info("📖 Last 5 rows in Google Sheets:")
            st.dataframe(pd.DataFrame(latest_rows))
        else:
            st.warning(f"⚠️ Could not read back from Google Sheets: {err2}")
    else:
        st.warning(f"⚠️ Could not save to Google Sheets: {err}")

    # Allow download as CSV
    st.download_button(
        "📥 Download Results as CSV",
        results_df.to_csv(index=False),
        "predictions.csv",
        "text/csv",
    )

    
    # === Manual entry path ===
    # -----------------------------
# UI — Manual entry path (single-column outcome)
# -----------------------------
st.subheader("📋 Enter Patient Data Manually")

# Define categorical (binary) features
categorical_features = [
    "HIV+", "def_Anemia", "R_Arth", "c_Pulm", "DM", "htn_C", "hypo_Thy",
    "liver_D", "Mets", "Obesity", "ren_Fail", "Tumor", "MI", "BA", "CVA",
    "ChroLiverDis", "Hemiplegia", "LapCholi", "OpenCholi", "Hernioplasty",
    "Herniotomy", "Lithotomy", "Pyeloplasty", "Appendicectomy", "Omentoplasty",
    "SmallBowelResection", "Laproscopic LysisOfAdhesions", "MRM",
    "Hysterectomy", "Prostectomy", "DiagLaprot", "Nephrectomy", "Gastrectomy",
    "Oesophagotomy", "UnimpDis_LAMA", "SuperficialSSI", "DeepSurgicalSSI",
    "OrganSpaceSSI", "Dehiscence", "GastricOutletObs", "GeneralisedPeritonitis",
    "pul_Complications", "c_Complication", "UTI", "Sepsis", "reoperation", "Readm"
]

# Ensure feature names are clean
feature_names = [f.strip() for f in feature_names]

with st.form("manual_form"):
    manual_data = {}
    for feature in feature_names:
        if feature in categorical_features:
            manual_data[feature] = st.selectbox(f"{feature}", ["No", "Yes"], index=0)
        else:
            manual_data[feature] = st.number_input(f"{feature}", step=0.1, value=0.0)

    submitted = st.form_submit_button("Predict")

if submitted:
    # Convert Yes/No → 1/0
    raw_input = manual_data.copy()
    for col in categorical_features:
        if col in raw_input:
            raw_input[col] = 1 if raw_input[col] == "Yes" else 0

    df_input = pd.DataFrame([raw_input])

    # Scale numeric features
    try:
        df_scaled = df_input.copy()
        numeric_to_scale = [
            col for col in scaler_features
            if col in df_scaled.columns and col not in categorical_features
        ]
        if numeric_to_scale:
            df_scaled[numeric_to_scale] = scaler.transform(df_scaled[numeric_to_scale])
        df_input = df_scaled
    except Exception as e:
        st.error(f"❌ Scaling error: {e}")
        st.stop()

    # Convert to array
    input_array = df_input.values

    # Predict ensemble
    all_probs = ensemble_models_predict_all(input_array)
    mean_probs = np.mean(all_probs, axis=0).reshape(-1)   # 1D
    mean_probs = ensure_single_output(mean_probs)
    std_devs = np.std(all_probs, axis=0).reshape(-1)
    entropy = np.atleast_1d(calculate_entropy(mean_probs))  # ensure array

    # Calibrate probabilities
    try:
        calibrated_probs = iso_reg.predict(mean_probs.reshape(-1, 1)).flatten()
    except Exception:
        calibrated_probs = iso_reg.predict(np.asarray(mean_probs).reshape(-1, 1)).flatten()

    predicted_labels = (calibrated_probs >= best_threshold).astype(int)

    # Display results
    st.subheader("Prediction Result")
    st.write(f"**Raw Probability:** {mean_probs[0]:.3f}")
    st.write(f"**Calibrated Probability:** {calibrated_probs[0]:.3f}")
    st.write(f"**Predicted Label:** {'High Risk' if predicted_labels[0] == 1 else 'Low Risk'}")
    st.write(f"**Uncertainty (Std Dev):** {std_devs[0]:.3f}")
    st.write(f"**Entropy:** {entropy[0]:.3f}")

    # Save raw input + prediction to Google Sheets
    row_to_save = raw_input.copy()
    row_to_save.update({
        "raw_probability": float(mean_probs[0]),
        "calibrated_probability": float(calibrated_probs[0]),
        "predicted_label": int(predicted_labels[0]),
        "std_deviation": float(std_devs[0]),
        "entropy": float(entropy[0]),
        "timestamp": pd.Timestamp.now().isoformat()
    })

    row_df = pd.DataFrame([row_to_save])
    ok, err = append_to_gsheet(row_df)
    if ok:
        st.success("✅ Saved prediction to Google Sheets.")

        # Show last 5 rows for verification
        latest_rows, err2 = read_from_gsheet(n=5)
        if latest_rows:
            st.info("📖 Last 5 rows in Google Sheets:")
            st.dataframe(pd.DataFrame(latest_rows))
        else:
            st.warning(f"⚠️ Could not read back from Google Sheets: {err2}")
    else:
        st.warning(f"⚠️ Could not save to Google Sheets: {err}")
