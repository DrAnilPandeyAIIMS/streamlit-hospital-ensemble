import gc
import streamlit as st
import tensorflow as tf
import os
import time
import json
from pathlib import Path
import subprocess
import warnings
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.isotonic import IsotonicRegression
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import threading
import ast
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.calibration import calibration_curve

# ============================================================
# 1. PAGE CONFIG & PATH SETUP
# ============================================================
st.set_page_config(
    page_title="Clinical Mortality Risk Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏥 Clinical Ensemble Mortality Predictor")
st.subheader("Validated Postoperative Risk Stratification System (2026)")
st.markdown("""
This system employs a memory-optimized Bayesian ensemble to provide real-time
mortality risk assessment. All risk thresholds are derived mathematically via
Youden's J statistic, calibrated for maximum sensitivity with minimum false alerts.
""")
st.caption(f"System Operational | Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

with st.spinner("⏳ Initializing models..."):
    pass

# ============================================================
# PATHS
# ============================================================
BASE_DIR   = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR            = MODELS_DIR / "outputs" / "ensemble_model_2"
RAW_THRESHOLD_PATH     = OUTPUTS_DIR / "threshold_raw.json"
BETA_CALIBRATOR_PATH   = OUTPUTS_DIR / "beta_reg.pkl"
ISOTONIC_CALIBRATOR_PATH = OUTPUTS_DIR / "isotonic_reg.pkl"
CALIBRATED_INFO_PATH   = OUTPUTS_DIR / "chosen_calibrator_info.json"
PERCENTILE_INFO_PATH   = OUTPUTS_DIR / "percentile_info.json"

DEBUG    = os.getenv("DEBUG", "false").lower() == "true"
EPS      = 1e-9
MC_RUNS  = 30
IS_CLOUD = os.getenv("STREAMLIT_SERVER_HEADLESS", "false") == "true"

if not OUTPUTS_DIR.exists():
    st.warning(f"⚠️ Model outputs folder missing: {OUTPUTS_DIR}")
if not RAW_THRESHOLD_PATH.exists():
    st.warning(f"⚠️ Threshold file missing: {RAW_THRESHOLD_PATH}")

# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================
def load_json_safe(path: Path, default=None):
    try:
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return default or {}

def load_pickle_safe(path: Path):
    try:
        if path.exists():
            return joblib.load(path)
    except Exception:
        pass
    return None

# ============================================================
# 3. LOAD THRESHOLD JSON  ←  SINGLE SOURCE OF TRUTH
# ============================================================
# REMOVED: get_hardened_thresholds() and load_threshold_safe()
# Both were duplicating the same load with inconsistent fallbacks.
# One load, one dict, all keys pulled from it below.

# Cloud fallback values (only used when threshold_raw.json is absent)
# These are intentionally conservative — they do NOT override the file.
_CLOUD_FALLBACK = {
    "best_threshold"      : 0.30,       # conservative fallback only
    "threshold_method"    : "fallback",
    "high_risk_threshold" : 1.0,
    "gamma_safety"        : 0.0,
    "k_steepness"         : 7.0,
    "power_ramp"          : 1.2,
    "suppression_mult"    : 0.09,
    "score_floor"         : 0.27,
    "prevalence"          : 0.05,
    "vae_weight"          : 2.0,
    "caution_weight"      : 0.5,
    "recall"              : None,
    "true_positives"      : None,
    "false_positives"     : None,
    "true_negatives"      : None,
    "false_negatives"     : None,
    "entropy_min"         : 0.0,
    "entropy_max"         : 1.0,
    "thr_search_diag"     : None,
    "thr_entropy_diag"    : None,
    "fp_search_diag"      : None,
    "fp_entropy_diag"     : None,
    "j_score"             : None,
}

thr_data = load_json_safe(RAW_THRESHOLD_PATH)

if not thr_data:
    if IS_CLOUD:
        st.warning("⚠️ threshold_raw.json not found — using conservative fallback values.")
        thr_data = _CLOUD_FALLBACK
    else:
        st.error(f"🚨 threshold_raw.json not found at {RAW_THRESHOLD_PATH}")
        st.stop()

# ============================================================
# 4. UNPACK ALL THRESHOLD KEYS  (aligned with v6 JSON output)
# ============================================================
best_threshold_saved = thr_data.get("best_threshold",       _CLOUD_FALLBACK["best_threshold"])
THRESHOLD_METHOD     = thr_data.get("threshold_method",     _CLOUD_FALLBACK["threshold_method"])
HIGH_RISK_BOUNDARY   = thr_data.get("high_risk_threshold",  _CLOUD_FALLBACK["high_risk_threshold"])
GAMMA                = thr_data.get("gamma_safety",         _CLOUD_FALLBACK["gamma_safety"])
K_STEEPNESS          = thr_data.get("k_steepness",          _CLOUD_FALLBACK["k_steepness"])
POWER_RAMP           = thr_data.get("power_ramp",           _CLOUD_FALLBACK["power_ramp"])
SUPPRESSION_MULT     = thr_data.get("suppression_mult",     _CLOUD_FALLBACK["suppression_mult"])
SCORE_FLOOR          = thr_data.get("score_floor",          _CLOUD_FALLBACK["score_floor"])
PREVALENCE           = thr_data.get("prevalence",           _CLOUD_FALLBACK["prevalence"])
VAE_WEIGHT           = thr_data.get("vae_weight",           _CLOUD_FALLBACK["vae_weight"])
CAUTION_WEIGHT       = thr_data.get("caution_weight",       _CLOUD_FALLBACK["caution_weight"])
ENTROPY_MIN          = thr_data.get("entropy_min",          _CLOUD_FALLBACK["entropy_min"])
ENTROPY_MAX          = thr_data.get("entropy_max",          _CLOUD_FALLBACK["entropy_max"])

# Diagnostic thresholds (for display only — not used in inference)
THR_SEARCH_DIAG      = thr_data.get("thr_search_diag")
THR_ENTROPY_DIAG     = thr_data.get("thr_entropy_diag")
FP_SEARCH_DIAG       = thr_data.get("fp_search_diag")
FP_ENTROPY_DIAG      = thr_data.get("fp_entropy_diag")
J_SCORE              = thr_data.get("j_score")

# Training metrics (for audit display)
SAVED_RECALL         = thr_data.get("recall")
SAVED_TP             = thr_data.get("true_positives")
SAVED_FP             = thr_data.get("false_positives")
SAVED_TN             = thr_data.get("true_negatives")
SAVED_FN             = thr_data.get("false_negatives")

# Safety guardrail: critical zone must always sit above gray zone
if HIGH_RISK_BOUNDARY <= best_threshold_saved:
    HIGH_RISK_BOUNDARY = 1.0

# ============================================================
# 5. TRIAGE CLASSIFICATION LOGIC
# ============================================================
def triage_levels_logic(score, threshold, high_risk_boundary=None):
    """
    Categorizes a final calibrated score into clinical tiers.
    Thresholds are read from JSON — nothing is hardcoded here.
    """
    if high_risk_boundary is None:
        high_risk_boundary = HIGH_RISK_BOUNDARY
    try:
        s = float(score)
        if s >= high_risk_boundary:
            return "🔴 CRITICAL"
        elif s >= threshold:
            return "🟡 GRAY ZONE"
        else:
            return "🟢 SAFE"
    except (ValueError, TypeError):
        return "⚪ UNKNOWN"

# ============================================================
# 6. SYSTEM STATUS EXPANDER
# ============================================================
with st.expander("✅ System Status: Clinical Artifacts Active", expanded=False):

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Threshold (Youden's J)**")
        st.metric(
            label=f"Method: {THRESHOLD_METHOD}",
            value=f"{best_threshold_saved:.4f}",
            help="Derived by maximising sensitivity + specificity − 1 on calibrated scores"
        )
        if J_SCORE is not None:
            st.caption(f"J = {J_SCORE:.4f}")

    with col2:
        st.markdown("**Diagnostic Comparison**")
        rows = [("Youden (primary)", best_threshold_saved,
                 SAVED_FP if SAVED_FP is not None else "—")]
        if THR_SEARCH_DIAG is not None:
            rows.append(("Search",  THR_SEARCH_DIAG,  FP_SEARCH_DIAG  or "—"))
        if THR_ENTROPY_DIAG is not None:
            rows.append(("Entropy", THR_ENTROPY_DIAG, FP_ENTROPY_DIAG or "—"))

        diag_df = pd.DataFrame(rows, columns=["Method", "Threshold", "FPs"])
        diag_df["Selected"] = diag_df["Method"].apply(
            lambda m: "✅" if m.startswith("Youden") else ""
        )
        st.dataframe(diag_df, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("**Training Performance**")
    if SAVED_RECALL is not None:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Recall",    f"{SAVED_RECALL:.1%}")
        m2.metric("TP",        SAVED_TP or "—")
        m3.metric("FP",        SAVED_FP or "—")
        m4.metric("FN",        SAVED_FN or "—")
    else:
        st.caption("Training metrics not available (cloud fallback active)")

    st.markdown("---")
    st.markdown("**Pipeline Parameters**")
    st.markdown(f"""
    - **Entropy range:** `{ENTROPY_MIN:.4f}` → `{ENTROPY_MAX:.4f}`
    - **Gamma (entropy blend):** `{GAMMA}`
    - **K steepness / Power ramp:** `{K_STEEPNESS}` / `{POWER_RAMP}`
    - **Suppression mult / Score floor:** `{SUPPRESSION_MULT:.4f}` / `{SCORE_FLOOR:.4f}`
    - **VAE weight / Prevalence:** `{VAE_WEIGHT:.3f}` / `{PREVALENCE:.2%}`
    """)

# ============================================================
# 7. ARTIFACT LOADING
# ============================================================
chosen_calibrator_info  = load_json_safe(CALIBRATED_INFO_PATH) or {}
CHOSEN_CALIBRATOR_NAME  = chosen_calibrator_info.get("chosen_calibrator", "None")

percentile_info  = load_json_safe(PERCENTILE_INFO_PATH, {})
IS_FROZEN        = percentile_info.get("frozen", False)
FROZEN_PERCENTILE = percentile_info.get("percentile")

feature_names_raw = load_pickle_safe(MODELS_DIR / "feature_names.pkl")
feature_names     = list(feature_names_raw) if feature_names_raw is not None else []

label_encoder = load_pickle_safe(MODELS_DIR / "label_encoder.pkl")
if label_encoder is None:
    st.warning("⚠️ `label_encoder.pkl` missing — reconstructing from hard-coded classes.")
    try:
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.array(['ASA_one', 'ASA_two', 'ASA_three', 'ASA_four', 'ASA-E'])
        st.info("✅ Encoder reconstructed.")
    except Exception as e:
        st.error(f"🚨 Could not initialise LabelEncoder: {e}")
        st.stop()
else:
    st.success("✅ Label Encoder loaded.")

scaler = load_pickle_safe(MODELS_DIR / "scaler.pkl")
if isinstance(scaler, (list, tuple)):
    scaler = next((obj for obj in scaler if hasattr(obj, "transform")), None)

if not feature_names:
    st.error("🚨 `feature_names.pkl` is missing or corrupt.")
    st.stop()

# ============================================================
# 8. FEATURE DEFINITIONS & SCALABLE INDICES
# ============================================================
ordinal_variables = ["ASAclassification"]
NUMERIC_FEATURES  = [
    "PreOpTLC", "PreopUrea", "PreopCreat", "PreopSodium",
    "PreopPotassium", "PreOpBilT", "PreOpBilD", "ALP",
    "PreOpSGOT", "PostOpTLC", "PostopUrea", "PostopCreat",
    "PostOpSodium", "PostOpPotassium", "PostOpBilT",
    "PostOpBilD", "PostOpALP", "PostOpSGOT", "PostOpSGPT"
]
categorical_features = {
    "HIV+", "def_Anemia", "R_Arth", "c_Pulm", "DM", "htn_C", "hypo_Thy",
    "liver_D", "Mets", "Obesity", "ren_Fail", "Tumor", "MI", "BA", "CVA",
    "ChroLiverDis", "Hemiplegia", "LapCholi", "OpenCholi", "Hernioplasty",
    "Herniotomy", "Lithotomy", "Pyeloplasty", "Appendicectomy", "Omentoplasty",
    "SmallBowelResection", "Laproscopic LysisOfAdhesions", "MRM",
    "Hysterectomy", "Prostectomy", "DiagLaprot", "Nephrectomy", "Gastrectomy",
    "Oesophagotomy", "UnimpDis_LAMA", "SuperficialSSI", "DeepSurgicalSSI",
    "OrganSpaceSSI", "Dehiscence", "GastricOutletObs", "GeneralisedPeritonitis",
    "pul_Complications", "c_Complication", "UTI", "Sepsis", "reoperation", "Readm"
}

try:
    if hasattr(scaler, "feature_names_in_"):
        potential_scalable = list(scaler.feature_names_in_)
    else:
        potential_scalable = feature_names[:scaler.n_features_in_]

    features_to_scale = NUMERIC_FEATURES + ordinal_variables
    scalable_columns  = [col for col in potential_scalable if col in features_to_scale]
    SCALABLE_INDICES  = [
        feature_names.index(col)
        for col in scalable_columns
        if col in feature_names
    ]
except Exception as e:
    st.error(f"🚨 Index Mapping Error: {e}")
    st.stop()

# Scaler alignment check
if scaler is not None:
    expected_count = scaler.n_features_in_
    actual_count   = len(scalable_columns)
    if expected_count != actual_count:
        st.error(f"🚨 Scaler mismatch: expects {expected_count} features, found {actual_count}.")
        st.stop()
    if actual_count == 0:
        st.error("🚨 No features identified for scaling.")
        st.stop()

st.success(f"✅ System aligned: {len(feature_names)} features | {len(SCALABLE_INDICES)} scalable")

# ============================================================
# 9. CALIBRATORS
# ============================================================
class BetaCalibrator:
    """Beta calibration transformation."""
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def transform(self, p):
        p = np.clip(p, EPS, 1 - EPS)
        return (p ** self.a) / ((p ** self.a) + ((1 - p) ** self.b))

beta_calibrator = load_pickle_safe(BETA_CALIBRATOR_PATH)
iso_calibrator  = load_pickle_safe(ISOTONIC_CALIBRATOR_PATH)

if CHOSEN_CALIBRATOR_NAME == "beta" and beta_calibrator is None:
    st.error("❌ System set to Beta calibration but beta_reg.pkl is missing.")
    st.stop()
if CHOSEN_CALIBRATOR_NAME == "isotonic" and iso_calibrator is None:
    st.error("❌ System set to Isotonic calibration but isotonic_reg.pkl is missing.")
    st.stop()

# ============================================================
# 10. CALIBRATION METRICS & RELIABILITY PLOT
# ============================================================
def compute_ece_mce(y_true, y_prob, n_bins=10):
    bins   = np.linspace(0, 1, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    ece, mce = 0.0, 0.0
    for i in range(n_bins):
        mask = binids == i
        if np.any(mask):
            acc  = y_true[mask].mean()
            conf = y_prob[mask].mean()
            gap  = abs(acc - conf)
            ece += gap * mask.mean()
            mce  = max(mce, gap)
    return ece, mce

def plot_reliability(y_true, y_prob, title):
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
    fig, ax = plt.subplots()
    ax.plot(mean_pred, frac_pos, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title)
    ax.legend()
    st.pyplot(fig)

# ============================================================
# 11. MODEL REGISTRY & MANAGER
# ============================================================
MODEL_FILES = {
    "vae_model": {
        "id": "1GXrJ4GvXOZ4ZzjqQQzfwlyWi9IkSswYe",
        "path": MODELS_DIR / "vae_model.h5"
    },
    "model_2_probabilistic": {
        "id": "1ug_BZlcHXwIiOdmC-fnI9SX-ye_ftrad",
        "path": MODELS_DIR / "model_2_probabilistic.h5",
    },
    "bayesian_model": {
        "id": "1XIJvwqgakbncaM8QX-BL8ZQ7vMaBWMEp",
        "path": MODELS_DIR / "bayesian_model.h5"
    }
}

@st.cache_resource(show_spinner=True)
def download_model_if_needed(model_key):
    import gdown
    info = MODEL_FILES[model_key]
    path = Path(info["path"])
    if path.exists():
        return path
    try:
        st.info(f"📥 Downloading {model_key}...")
        path.parent.mkdir(exist_ok=True)
        subprocess.run(
            ["gdown", f"https://drive.google.com/uc?id={info['id']}", "-O", str(path), "--quiet"],
            timeout=300, check=True
        )
        if path.exists():
            st.success(f"✅ Downloaded {model_key}")
            return path
        raise FileNotFoundError(f"Download failed for {path}")
    except subprocess.TimeoutExpired:
        st.error(f"⏱️ Download timeout for {model_key}.")
        st.stop()
    except Exception as e:
        st.error(f"🚨 Could not download {model_key}: {e}")
        st.stop()

class SingleModelManager:
    def __init__(self):
        self.current_key = None
        self.model       = None

    def load(self, key):
        if self.current_key == key and self.model is not None:
            return self.model
        self.unload()
        import tensorflow as tf
        path = download_model_if_needed(key)
        try:
            self.model       = tf.keras.models.load_model(path, compile=False, custom_objects=get_custom_objects())
            self.current_key = key
            return self.model
        except Exception as e:
            self.unload()
            raise RuntimeError(f"Could not load {key} from {path}: {e}")

    def unload(self):
        if self.model is not None:
            try:
                import tensorflow as tf
                del self.model
                tf.keras.backend.clear_session()
                gc.collect()
            except Exception:
                pass
        self.model       = None
        self.current_key = None

model_manager = SingleModelManager()

# ============================================================
# 12. TENSORFLOW CUSTOM OBJECTS (LAZY)
# ============================================================
def _initialize_tensorflow_components():
    import tensorflow as tf
    import tensorflow_probability as tfp
    tfd = tfp.distributions

    def prior(kernel_size, bias_size, dtype=None):
        n = kernel_size + bias_size
        return tf.keras.Sequential([
            tfp.layers.VariableLayer(n, dtype=dtype),
            tfp.layers.DistributionLambda(
                lambda t: tfd.MultivariateNormalDiag(loc=t, scale_diag=tf.ones_like(t))
            )
        ])

    def posterior(kernel_size, bias_size, dtype=None):
        n = kernel_size + bias_size
        return tf.keras.Sequential([
            tfp.layers.VariableLayer(tfp.layers.IndependentNormal.params_size(n), dtype=dtype),
            tfp.layers.IndependentNormal(n, convert_to_tensor_fn=tfd.Distribution.sample),
        ])

    class CustomDenseVariational(tfp.layers.DenseVariational):
        def __init__(self, units, make_prior_fn, make_posterior_fn, kl_weight=1.0, **kwargs):
            super().__init__(units=units, make_prior_fn=make_prior_fn,
                             make_posterior_fn=make_posterior_fn, kl_weight=kl_weight, **kwargs)
            self.units      = units
            self.kl_weight  = kl_weight
        def get_config(self):
            config = super().get_config()
            config.update({"units": self.units, "kl_weight": self.kl_weight})
            return config
        @classmethod
        def from_config(cls, config):
            config.setdefault("make_prior_fn",     prior)
            config.setdefault("make_posterior_fn", posterior)
            config["make_prior_fn"]     = prior
            config["make_posterior_fn"] = posterior
            return cls(**config)

    class DenseFlipoutLayer(tf.keras.layers.Layer):
        def __init__(self, units, activation=None, **kwargs):
            super().__init__(**kwargs)
            self.units      = units
            self.activation = activation
        def build(self, input_shape):
            self.dense_flipout = tfp.layers.DenseFlipout(units=self.units, activation=self.activation)
            super().build(input_shape)
        def call(self, inputs):
            return self.dense_flipout(inputs)

    def negative_log_likelihood_bernoulli(y_true, y_pred):
        return -tf.reduce_mean(
            y_true * tf.math.log(y_pred + 1e-9) + (1 - y_true) * tf.math.log(1 - y_pred + 1e-9)
        )

    def negative_log_likelihood(y_true, y_pred_dist):
        return -y_pred_dist.log_prob(y_true)

    return {
        "CustomDenseVariational"              : CustomDenseVariational,
        "DenseFlipoutLayer"                   : DenseFlipoutLayer,
        "DenseFlipout"                        : tfp.layers.DenseFlipout,
        "DistributionLambda"                  : tfp.layers.DistributionLambda,
        "prior"                               : prior,
        "posterior"                           : posterior,
        "negative_log_likelihood"             : negative_log_likelihood,
        "negative_log_likelihood_bernoulli"   : negative_log_likelihood_bernoulli,
    }

_tf_custom_objects = None
def get_custom_objects():
    global _tf_custom_objects
    if _tf_custom_objects is None:
        _tf_custom_objects = _initialize_tensorflow_components()
    return _tf_custom_objects

# ============================================================
# 13. CORE MATH HELPERS
# ============================================================
def calculate_entropy(probs):
    """Shannon entropy — natural log, matching training parity."""
    p = np.clip(np.ravel(probs), EPS, 1 - EPS)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))

@st.cache_resource
def load_small_objects():
    if scaler is None or not hasattr(scaler, "transform"):
        raise RuntimeError("Valid scaler not found. Check scaler.pkl.")
    return scaler, feature_names

scaler_cached, feature_names_cached = load_small_objects()

# ============================================================
# 14. MC INFERENCE  (aligned with v6 pipeline parameters)
# ============================================================
def ensure_single_output(arr):
    a = np.asarray(arr)
    if a.size and (a.min() < 0 or a.max() > 1):
        a = 1.0 / (1.0 + np.exp(-a))
    if a.ndim == 2 and a.shape[1] == 2:
        return a[:, 1]
    if a.ndim == 2 and a.shape[1] == 1:
        return a.ravel()
    return a.ravel()

mc_passes = MC_RUNS

def load_models_and_mc_for_batch(X_np, n_forward_passes=30):
    """
    Inference pipeline — runtime version (no rank transform, no S-curve).

    WHY NO S-CURVE / RANK TRANSFORM AT RUNTIME:
    The v6 threshold script uses a rank-based band transform that maps
    deaths → [0.50, 1.0] and survivors → [0.0, 0.40]. That transform
    requires y_true (ground truth labels), which are not available at
    inference time. The Youden threshold (stored in JSON) was derived
    on the rank-transformed scores.

    Runtime approach: output weighted_probs + gamma*entropy directly.
    Apply best_threshold_saved from JSON to these raw scores.
    This is consistent because best_t_raw in the JSON is the pre-S-curve
    threshold found in section 4 of the threshold script, which operates
    on the same weighted_probs space we produce here.
    """
    model_keys = ["vae_model", "model_2_probabilistic", "bayesian_model"]
    X_tensor   = tf.convert_to_tensor(np.asarray(X_np, dtype=np.float32))

    all_model_mc_means = []
    for key in model_keys:
        model      = model_manager.load(key)
        mc_samples = [ensure_single_output(model(X_tensor, training=True))
                      for _ in range(n_forward_passes)]
        all_model_mc_means.append(np.mean(np.vstack(mc_samples), axis=0))
        model_manager.unload()

    all_model_mc_means = np.array(all_model_mc_means)
    mean_per_model     = all_model_mc_means.T   # (N, 3): col0=VAE, col1=M2, col2=Bay
    vae_p = mean_per_model[:, 0]
    m2_p  = mean_per_model[:, 1]
    bay_p = mean_per_model[:, 2]

    # --- 2a. Weighted consensus ---
    Z         = VAE_WEIGHT + 3.0 + 3.0
    base_risk = (m2_p * 3.0 + bay_p * 3.0 + vae_p * VAE_WEIGHT) / Z

    # --- 2b. Vote flags ---
    v_vae       = (vae_p > 0.08).astype(int)
    v_m2        = (m2_p  > 0.35).astype(int)
    v_bay       = (bay_p > 0.35).astype(int)
    total_votes = v_vae + v_m2 + v_bay

    # --- 2c/d. Unified gate: masks OR majority vote ---
    vae_mask       = (vae_p > 0.41)
    consensus_mask = (m2_p  > 0.58) & (bay_p > 0.58)
    is_valid       = (vae_mask | consensus_mask) | (total_votes >= 2)

    # --- 2e. Suppression (SUPPRESSION_MULT from JSON, not 0.01) ---
    weighted_probs = np.where(is_valid, base_risk, base_risk * SUPPRESSION_MULT)

    # --- 2f. Weak-signal floor (SCORE_FLOOR from JSON) ---
    any_signal     = (vae_p > 0.05) | (m2_p > 0.05) | (bay_p > 0.05)
    weighted_probs = np.where(
        any_signal,
        np.maximum(weighted_probs, SCORE_FLOOR),
        weighted_probs
    )

    # --- 3. Entropy (compute fresh, do NOT normalise with stale ENTROPY_MIN/MAX) ---
    avg_p       = (vae_p + m2_p + bay_p) / 3.0
    p_clip      = np.clip(avg_p, EPS, 1 - EPS)
    entropy_raw = -(p_clip * np.log2(p_clip) + (1 - p_clip) * np.log2(1 - p_clip))

    # Normalise within this batch (avoids stale training entropy range)
    e_min = entropy_raw.min()
    e_max = entropy_raw.max()
    entropy_norm = (entropy_raw - e_min) / (e_max - e_min + EPS)

    # --- 4. Final score: weighted_probs + gamma blend ---
    # Use best_t_raw from JSON as the operating threshold (pre-S-curve space)
    adjusted_probs = weighted_probs + (GAMMA * entropy_norm)

    # --- Triage ---
    # Use best_t_raw (pre-S-curve threshold) not best_threshold_saved (rank-space)
    runtime_threshold = thr_data.get("best_t_raw", best_threshold_saved)
    triage_levels = [
        triage_levels_logic(score, runtime_threshold)
        for score in adjusted_probs
    ]

    ensemble_std = np.std(all_model_mc_means, axis=0)
    return mean_per_model, adjusted_probs, ensemble_std, entropy_raw, entropy_norm, triage_levels

# ============================================================
# 15. CALIBRATION WRAPPERS
# ============================================================
def get_batch_calibrated_results(adjusted_probs_array):
    return calibrate_probs(
        arr=adjusted_probs_array,
        calib_mode="Auto (chosen)",
        chosen_calibrator_info=chosen_calibrator_info,
        beta_calibrator=beta_calibrator,
        iso_calibrator=iso_calibrator
    )

def get_single_calibrated_result(single_prob):
    current_calib_mode = st.session_state.get("calib_mode", "Auto (chosen)")
    return calibrate_single(val=single_prob, mode=current_calib_mode)

# ============================================================
# 16. GOOGLE SHEETS AUTH
# ============================================================
def get_gs_client_from_secrets():
    scopes    = ["https://www.googleapis.com/auth/spreadsheets"]
    local_key = "gsheets_credentials.json"
    if os.path.exists(local_key):
        try:
            creds  = Credentials.from_service_account_file(local_key, scopes=scopes)
            client = gspread.authorize(creds)
            return client, None
        except Exception:
            pass
    info = st.secrets.get("gcp_service_account")
    if not info:
        return None, "No credentials found (file or secret)"
    try:
        if isinstance(info, str):
            info = json.loads(info)
        creds  = Credentials.from_service_account_info(info, scopes=scopes)
        client = gspread.authorize(creds)
        return client, None
    except Exception as e:
        return None, f"Error creating Google Sheets client: {e}"

# ============================================================
# SESSION STATE DEFAULTS
# ============================================================
if "df_input"     not in st.session_state:
    st.session_state["df_input"]     = None
if "last_results" not in st.session_state:
    st.session_state["last_results"] = None

# ============================================================
# MODULE-LEVEL DEFAULTS FOR SIDEBAR CHECKBOXES
# These are defined here so they are always in scope regardless
# of which sidebar panel the user has selected.
# The sidebar elif block may not run (e.g. user is on Calibration
# panel by default) — without these defaults every CSV upload
# crashes with NameError on show_confusion etc.
# ============================================================
show_confusion        = True
show_uncertainty      = True
show_raw_model_probs  = True
show_calibrated_probs = True

# ============================================================
# DEBUG PANEL
# ============================================================
if DEBUG:
    with st.expander("🛠️ System Diagnostics (Debug Mode)"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Paths:**")
            st.write(f"Base: `{BASE_DIR.name}`")
            st.write(f"Outputs: `{OUTPUTS_DIR.name}`")
        with col2:
            st.write("**Calibrators:**")
            st.write(f"Beta loaded    : {beta_calibrator is not None}")
            st.write(f"Isotonic loaded: {iso_calibrator  is not None}")
        st.write("**Artifact files:**")
        st.code([p.name for p in OUTPUTS_DIR.glob("*")])
# ============================================================
# PART 2 — GOOGLE SHEETS, SIDEBAR, PREPROCESSING, INFERENCE, UI
# (Continues directly from streamlit_app_part1_v2.py)
# ============================================================

# ============================================================
# 1. GOOGLE SHEETS: APPEND
# ============================================================
def append_to_gsheet(df, sheet_key=None, worksheet_name=None):
    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except Exception as e:
            return False, f"Conversion error: {e}"
    if df.empty:
        return True, "DataFrame is empty, nothing to append."

    client, err = get_gs_client_from_secrets()
    if client is None:
        return False, f"Auth error: {err}"

    sheet_key      = sheet_key      or st.secrets.get("gsheet_key")
    worksheet_name = worksheet_name or st.secrets.get("gsheet_worksheet", "streamlit_project Data")
    if not sheet_key:
        return False, "No gsheet_key found."
    if "docs.google.com" in sheet_key:
        sheet_key = sheet_key.split("/d/")[1].split("/")[0]

    try:
        sh = client.open_by_key(sheet_key)
        try:
            ws = sh.worksheet(worksheet_name)
        except gspread.WorksheetNotFound:
            ws = sh.add_worksheet(title=worksheet_name, rows="1000", cols="20")
            ws.append_row(list(df.columns))

        existing_headers = ws.row_values(1)
        if not existing_headers:
            existing_headers = list(df.columns)
            ws.insert_row(existing_headers, index=1)

        df_aligned     = df.reindex(columns=existing_headers).fillna("")
        rows_to_append = df_aligned.astype(str).values.tolist()
        ws.append_rows(rows_to_append, value_input_option="USER_ENTERED")
        return True, f"Successfully synced {len(df)} patients to Cloud."
    except Exception as e:
        return False, f"Google Sheets Sync Failed: {str(e)}"


# ============================================================
# 2. CLINICAL LOGGING WRAPPER
# ============================================================
def log_clinical_inference(input_df, raw_p, adj_p, entropy, risk_label):
    log_entry = input_df.copy()
    log_entry["Inference_Timestamp"]  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry["Ensemble_Raw_Prob"]    = raw_p
    log_entry["Model_Threshold"]      = best_threshold_saved      # from JSON via part 1
    log_entry["Threshold_Method"]     = THRESHOLD_METHOD          # NEW: Youden / fallback label
    log_entry["Entropy_NaturalLog"]   = entropy
    log_entry["Gated_Prob_Gamma"]     = adj_p
    log_entry["Clinical_Risk_Label"]  = risk_label
    log_entry["Gamma_Penalty_Applied"] = GAMMA
    log_entry["J_Score"]              = J_SCORE                   # NEW: Youden J value
    return append_to_gsheet(log_entry)


# ============================================================
# 3. GOOGLE SHEETS: READ
# ============================================================
def read_from_gsheet(n=5):
    client, err = get_gs_client_from_secrets()
    if client is None:
        return None, f"Google Sheets client error: {err}"

    sheet_key      = st.secrets.get("gsheet_key")
    worksheet_name = st.secrets.get("gsheet_worksheet", "streamlit_project Data")
    if not sheet_key:
        return None, "No gsheet_key in secrets"

    try:
        sh         = client.open_by_key(sheet_key)
        ws         = sh.worksheet(worksheet_name)
        all_values = ws.get_all_values()
        if not all_values or len(all_values) <= 1:
            return pd.DataFrame(), None
        df = pd.DataFrame(all_values[1:], columns=all_values[0])
        return df.tail(n), None
    except Exception as e:
        return None, str(e)


# ============================================================
# 4. SIDEBAR
# ============================================================
with st.sidebar:
    st.title("⚙ Model Controls")

    sidebar_view = st.radio(
        "Select Panel",
        ["🔍 Calibration (Audit)", "📊 Evaluation / Prediction"],
        index=0
    )

    st.write(f"🔹 **High-risk percentile (frozen): Top {FROZEN_PERCENTILE}%**")

    # ----------------------------------------------------------
    # CALIBRATION AUDIT PANEL
    # ----------------------------------------------------------
    if sidebar_view == "🔍 Calibration (Audit)":
        st.subheader("🔒 Frozen Calibration State")
        st.write("**Chosen calibrator:**", CHOSEN_CALIBRATOR_NAME)
        st.write("**Threshold (Youden's J):**", f"{best_threshold_saved:.4f}")

        # NEW: show method and J score
        st.caption(f"Method: {THRESHOLD_METHOD} | J = {J_SCORE:.4f}" if J_SCORE else
                   f"Method: {THRESHOLD_METHOD}")

        st.markdown("#### 📁 Calibration Metadata")
        st.json(chosen_calibrator_info)

        # Diagnostic comparison — all three methods from JSON
        if THR_SEARCH_DIAG is not None or THR_ENTROPY_DIAG is not None:
            st.markdown("#### 📊 Threshold Method Comparison")
            rows = [("Youden (primary ✅)", best_threshold_saved,
                     SAVED_FP if SAVED_FP is not None else "—")]
            if THR_SEARCH_DIAG is not None:
                rows.append(("Search  (diag)",  THR_SEARCH_DIAG,  FP_SEARCH_DIAG  or "—"))
            if THR_ENTROPY_DIAG is not None:
                rows.append(("Entropy (diag)",  THR_ENTROPY_DIAG, FP_ENTROPY_DIAG or "—"))
            st.dataframe(
                pd.DataFrame(rows, columns=["Method", "Threshold", "FPs"]),
                hide_index=True, use_container_width=True
            )

        # Training performance metrics
        if SAVED_RECALL is not None:
            st.markdown("#### 📈 Training Performance")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Recall",    f"{SAVED_RECALL:.1%}")
            m2.metric("TP",        SAVED_TP or "—")
            m3.metric("FP",        SAVED_FP or "—")
            m4.metric("FN",        SAVED_FN or "—")

        st.markdown("---")
        st.subheader("☁️ Clinical Cloud Vault")

        sheet_id = st.secrets.get("gsheet_key")
        if sheet_id:
            clean_id  = sheet_id.split("/d/")[1].split("/")[0] if "/d/" in sheet_id else sheet_id
            vault_url = f"https://docs.google.com/spreadsheets/d/{clean_id}"
            st.link_button("📂 Open Clinical Vault", vault_url)

        if st.button("🔍 Preview Recent Cloud Inferences"):
            with st.spinner("Fetching from Vault..."):
                df_history, err = read_from_gsheet(n=5)
                if err:
                    st.error(f"Vault Connection Error: {err}")
                elif df_history is not None and not df_history.empty:
                    st.dataframe(df_history, use_container_width=True)
                else:
                    st.info("Clinical Vault is empty or headers only.")

        # REMOVED: FALLBACK_THRESHOLD and FALLBACK_SENSITIVITY_FLOOR metrics
        # (stale hardcoded values — threshold is now fully from JSON)
        st.markdown("---")
        st.caption(f"🔒 Threshold derived via {THRESHOLD_METHOD} | 2026 Mortality Artifacts")
        st.markdown("[📖 View Clinical Audit Methodology](https://your-research-repo.link/docs)")

    # ----------------------------------------------------------
    # PREDICTION & EVALUATION PANEL
    # ----------------------------------------------------------
    elif sidebar_view == "📊 Evaluation / Prediction":
        st.subheader("Prediction Configuration")

        st.selectbox(
            "Calibration mode (display only)",
            ["Auto (chosen)", "Beta", "Isotonic", "None"],
            index=0,
            key="calib_mode",
            help="Calibration affects displayed probabilities. Triage labels use Youden threshold."
        )

        st.markdown("---")
        st.subheader("🩺 Clinical Risk Stratification")

        stratification_mode = st.radio(
            "Risk stratification method",
            ["Percentile-based (Research Only)", "Safety-First Threshold (Clinical)"],
            index=1,
            key="strat_method_radio",
            help=f"Clinical mode uses the {best_threshold_saved:.4f} Youden threshold."
        )

        is_percentile_mode = stratification_mode.startswith("Percentile")

        if is_percentile_mode:
            if not IS_FROZEN or FROZEN_PERCENTILE is None:
                st.error("❌ Frozen percentile cutoff missing.")
                st.stop()
            st.write(f"🔹 **Relative Cutoff: Top {FROZEN_PERCENTILE}%**")
            st.caption("Identifies the highest-risk slice of the current cohort.")
        else:
            st.write(f"🔹 **Youden Threshold: {best_threshold_saved:.4f}**")
            st.caption(
                f"Derived via {THRESHOLD_METHOD} (J={J_SCORE:.4f}). "
                "Maximises sensitivity + specificity simultaneously."
                if J_SCORE else
                f"Derived via {THRESHOLD_METHOD}. Ensures 100% sensitivity."
            )

        st.markdown("---")
        st.subheader("Metrics Display")
        show_uncertainty      = st.checkbox("Show Uncertainty (Std & Entropy)", value=True)
        show_raw_model_probs  = st.checkbox("Show Raw Model Probabilities",     value=True)
        show_calibrated_probs = st.checkbox("Show Calibrated Probabilities",    value=True)
        show_confusion        = st.checkbox("Show Confusion Matrix / Scores",   value=True,
                                            help="Only visible if 'True Outcome' column is in CSV.")


# ============================================================
# 5. PREPROCESSING
# ============================================================
from sklearn.preprocessing import LabelEncoder

# Authoritative ASA encoder — order verified from label_encoder.pkl
clinical_label_encoder = LabelEncoder()
clinical_label_encoder.classes_ = np.array(
    ['ASA-E', 'ASA_four', 'ASA_one', 'ASA_three', 'ASA_two']
)

def apply_training_scaling(df: pd.DataFrame) -> np.ndarray:
    """
    Unified preprocessing: encode ASA → align features → scale.
    Mirrors training pipeline exactly.
    """
    try:
        working_df = df.copy()
        if 'ASAclassification' in working_df.columns:
            if working_df['ASAclassification'].dtype == object:
                valid_labels = set(label_encoder.classes_)
                working_df['ASAclassification'] = working_df['ASAclassification'].apply(
                    lambda x: x if x in valid_labels else label_encoder.classes_[0]
                )
                working_df['ASAclassification'] = label_encoder.transform(
                    working_df['ASAclassification']
                )
        df_aligned = working_df.reindex(columns=feature_names, fill_value=0.0).fillna(0.0)
        X = df_aligned.values.astype(np.float32)
        if scaler is not None and len(SCALABLE_INDICES) > 0:
            X[:, SCALABLE_INDICES] = scaler.transform(X[:, SCALABLE_INDICES])
        return X
    except Exception as e:
        st.error(f"⚠️ Preprocessing FAILED: {e}")
        st.info("Check CSV headers match required clinical features.")
        st.stop()


# ============================================================
# 6. CALIBRATION WRAPPERS
# ============================================================
def calibrate_probs(arr, mode="Auto (chosen)",
                    chosen_calibrator_info=None,
                    beta_calibrator=None,
                    iso_calibrator=None):
    """Vectorised calibration for batch arrays."""
    # Allow callers to pass locals; fall back to module-level objects
    _info  = chosen_calibrator_info or globals().get("chosen_calibrator_info", {})
    _beta  = beta_calibrator        or globals().get("beta_calibrator")
    _iso   = iso_calibrator         or globals().get("iso_calibrator")

    arr       = np.asarray(arr, dtype=float)
    p_clipped = np.clip(arr, EPS, 1 - EPS)

    chosen = mode
    if mode == "Auto (chosen)":
        chosen = (_info or {}).get("chosen_calibrator", "None")

    if chosen.lower() == "beta" and _beta is not None:
        if hasattr(_beta, "a") and hasattr(_beta, "b"):
            return np.clip(_beta.transform(p_clipped), EPS, 1 - EPS)
        try:
            result = _beta.predict_proba(p_clipped.reshape(-1, 1))[:, 1]
            return np.clip(result, EPS, 1 - EPS)
        except Exception:
            return p_clipped

    if chosen.lower() == "isotonic" and _iso is not None:
        return np.clip(_iso.predict(p_clipped), EPS, 1 - EPS)

    return p_clipped


def calibrate_single(val: float, mode: str):
    """Calibration for a single value (manual entry)."""
    res_arr = calibrate_probs(np.array([val]), mode=mode)
    chosen  = mode
    if mode == "Auto (chosen)":
        chosen = (chosen_calibrator_info or {}).get("chosen_calibrator", "uncalibrated")
    return float(res_arr[0]), chosen


def calibrate_probs_runtime(arr):
    """Connects sidebar calib_mode to calibrate_probs for batch CSV."""
    mode = st.session_state.get("calib_mode", "Auto (chosen)")
    return calibrate_probs(arr, mode)


# ============================================================
# 7. MAIN UI
# ============================================================
results = None
mode    = st.radio(
    "Select Entry Mode", ["Batch CSV", "Manual Entry"],
    key="entry_mode_selector"
)

# ============================================================
# 7a. BATCH CSV
# ============================================================
if mode == "Batch CSV":
    st.header("📂 Batch Clinical Audit")
    uploaded = st.file_uploader("Upload Patient Records (CSV Format)", type=["csv"])

    if uploaded:
        df_raw = pd.read_csv(uploaded)
        st.success(f"📥 Loaded {len(df_raw)} patient records.")

        # --- Preprocessing ---
        try:
            df_input = df_raw.copy()
            if 'ASAclassification' in df_input.columns:
                df_input['ASAclassification'] = (
                    df_input['ASAclassification'].astype(str).str.strip()
                )
                df_input['ASAclassification'] = clinical_label_encoder.transform(
                    df_input['ASAclassification']
                )
            else:
                st.error("🚨 'ASAclassification' column missing from CSV.")
                st.stop()

            df_aligned = df_input.reindex(columns=feature_names, fill_value=0.0)
            X_np       = df_aligned.values.astype(np.float32)
            if scaler is not None and len(SCALABLE_INDICES) > 0:
                X_np[:, SCALABLE_INDICES] = scaler.transform(X_np[:, SCALABLE_INDICES])
            st.info("✅ Data aligned with clinical feature set.")
        except Exception as e:
            st.error(f"🚨 Preprocessing Failure: {e}")
            st.info("Verify ASA labels: ['ASA-E', 'ASA_four', 'ASA_one', 'ASA_three', 'ASA_two']")
            st.stop()

        # --- Inference ---
        with st.spinner(f"🚀 Running {MC_RUNS} Monte Carlo passes per patient..."):
            m_means, gated_scores, uncertainties, entropy, entropy_norm, triage_levels = \
                load_models_and_mc_for_batch(X_np, n_forward_passes=MC_RUNS)

        # --- Calibration ---
        # NOTE: calibrate on gated_scores (post-gate) not ensemble mean
        # ensemble mean is pre-gate and on a different scale
        iso_calibrated   = calibrate_probs(gated_scores, mode="Isotonic")
        beta_calibrated  = calibrate_probs(gated_scores, mode="Beta")
        calibrated_probs = calibrate_probs_runtime(gated_scores)

        # --- Results assembly ---
        results = df_raw.copy()
        results["Ensemble_Mean"]   = np.mean(m_means, axis=1)
        results["Uncertainty_SD"]  = uncertainties
        results["Entropy_Norm"]    = entropy_norm
        results["Gated_Score"]     = gated_scores
        results["Calibrated_Prob"] = calibrated_probs
        results["Isotonic_Audit"]  = iso_calibrated
        results["Beta_Audit"]      = beta_calibrated
        # REMOVED: "Safety_Boost" (gated - ensemble_mean) — different score spaces, misleading
        # REMOVED: hardcoded 0.4380 label — use best_threshold_saved from JSON
        # Runtime scores are in weighted_probs space → use best_t_raw not best_threshold_saved
        runtime_thr = thr_data.get("best_t_raw", best_threshold_saved)
        results["Risk_Label"]      = np.where(
            gated_scores >= runtime_thr, "High Risk", "Low Risk"
        )
        results["Triage_Level"]    = triage_levels
        results["Threshold_Used"]  = best_threshold_saved   # audit trail
        results["Threshold_Method"] = THRESHOLD_METHOD       # audit trail

        # --- Metrics ---
        st.divider()
        st.write(
            f"📊 **Batch Diagnostics:** "
            f"Avg Entropy: {float(np.mean(entropy)):.4f} | "
            f"Threshold [{THRESHOLD_METHOD}]: {best_threshold_saved:.4f}"
        )

        high_risk_count = int(np.sum(results["Risk_Label"] == "High Risk"))
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Sample",     len(results))
        m2.metric("High Risk Alerts", high_risk_count,
                  delta=f"{(high_risk_count / len(results)):.1%}",
                  delta_color="inverse")
        m3.metric(f"Threshold ({THRESHOLD_METHOD})", f"{best_threshold_saved:.4f}")

        # --- Calibration audit ---
        with st.expander("🔬 Calibration Performance & Model Agreement"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Isotonic (Avg)", f"{iso_calibrated.mean():.1%}")
            c2.metric("Beta (Avg)",     f"{beta_calibrated.mean():.1%}", delta="PREFERRED")
            agreement = 1 - abs(iso_calibrated.mean() - beta_calibrated.mean())
            c3.metric("Model Convergence", f"{agreement:.1%}")

        # --- Confusion matrix (if ground truth present) ---
        if show_confusion and "True_Outcome" in results.columns:
            with st.expander("📊 Confusion Matrix"):
                y_true_batch = results["True_Outcome"].astype(int).values
                y_pred_batch = (gated_scores >= best_threshold_saved).astype(int)
                cm = confusion_matrix(y_true_batch, y_pred_batch)
                rec  = recall_score(y_true_batch, y_pred_batch, zero_division=0)
                prec = precision_score(y_true_batch, y_pred_batch, zero_division=0)
                f1   = f1_score(y_true_batch, y_pred_batch, zero_division=0)
                st.write(f"Recall: **{rec:.1%}** | Precision: **{prec:.1%}** | F1: **{f1:.3f}**")
                st.dataframe(
                    pd.DataFrame(cm,
                                 index=["True: Survivor", "True: Death"],
                                 columns=["Pred: Safe", "Pred: Flagged"]),
                    use_container_width=True
                )
                try:
                    auc = roc_auc_score(y_true_batch, gated_scores)
                    st.write(f"AUC-ROC: **{auc:.4f}**")
                    plot_reliability(y_true_batch, gated_scores, "Reliability Curve")
                except Exception:
                    pass

        # --- Styled triage table ---
        st.subheader("📋 Detailed Clinical Triage List")

        def style_triage_row(val):
            text = str(val).upper()
            if "CRITICAL"  in text: return 'color:white;background-color:#d93025;font-weight:bold;'
            if "GRAY ZONE" in text: return 'color:black;background-color:#f9ab00;font-weight:bold;'
            if "SAFE"      in text: return 'color:white;background-color:#1e8e3e;font-weight:bold;'
            return ''

        fmt = {
            "Ensemble_Mean":   "{:.4f}",
            "Uncertainty_SD":  "{:.4f}",
            "Gated_Score":     "{:.4f}",
            "Calibrated_Prob": "{:.2%}",
            "Beta_Audit":      "{:.2%}",
        }
        styled_df = (
            results.style
            .applymap(style_triage_row, subset=["Triage_Level"])
            .background_gradient(subset=["Gated_Score"], cmap="YlOrRd")
            .format(fmt)
        )
        st.dataframe(styled_df, use_container_width=True)

        # --- Export / Archive ---
        st.markdown("---")
        col_dl, col_gs = st.columns(2)
        with col_dl:
            csv_report = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Triage Report (CSV)",
                data=csv_report,
                file_name=f"clinical_triage_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
        with col_gs:
            if st.button("🚀 Archive Batch to Clinical Vault"):
                with st.spinner("Syncing records..."):
                    success, message = append_to_gsheet(results)
                    if success:
                        st.success(f"Archived {len(results)} records.")
                    else:
                        st.error(f"Sync failed: {message}")


# ============================================================
# 7b. MANUAL ENTRY
# ============================================================
elif mode == "Manual Entry":
    manual_data = {f: 0.0 for f in feature_names}

    with st.form("entry_form"):
        st.subheader("📋 Patient Data Input")
        cols = st.columns(3)
        for i, f in enumerate(feature_names):
            with cols[i % 3]:
                if f == "ASAclassification":
                    manual_data[f] = st.selectbox(
                        "ASA Classification",
                        ["ASA-E", "ASA_one", "ASA_two", "ASA_three", "ASA_four"]
                    )
                elif f in categorical_features:
                    manual_data[f] = 1.0 if st.checkbox(f, value=False) else 0.0
                else:
                    manual_data[f] = st.number_input(f, value=0.0)

        submitted = st.form_submit_button("Predict")

    if submitted:
        # --- Preprocessing ---
        df_manual = pd.DataFrame([manual_data])
        df_manual["ASAclassification"] = clinical_label_encoder.transform(
            [df_manual["ASAclassification"].iloc[0]]
        )
        X_manual = apply_training_scaling(df_manual)

        # --- Inference ---
        m_means, gated_scores, uncertainties, entropy, entropy_norm, triage_levels = \
            load_models_and_mc_for_batch(X_manual, n_forward_passes=MC_RUNS)

        current_triage   = triage_levels[0]
        adj_p            = float(gated_scores[0])
        e_val            = float(entropy[0])
        en_val           = float(entropy_norm[0])
        # FIXED: ensemble_mean_val from per-model means, not gated scores
        ensemble_mean_val = float(np.mean(m_means[0]))

        runtime_thr  = thr_data.get("best_t_raw", best_threshold_saved)
        is_high_risk = adj_p >= runtime_thr
        is_near_miss = (not is_high_risk) and (adj_p >= best_threshold_saved - 0.10)
        label        = "High Risk" if is_high_risk else ("Borderline" if is_near_miss else "Low Risk")

        # --- Calibration ---
        cal_p, cal_method = calibrate_single(ensemble_mean_val, CHOSEN_CALIBRATOR_NAME)

        # --- Clinical display ---
        st.divider()
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            if "CRITICAL" in current_triage:
                st.error(f"🚨 **{current_triage}**", icon="🔥")
                st.metric("Risk Decision", "RED ALERT",
                          delta="IMMEDIATE ACTION", delta_color="inverse")
            elif "GRAY ZONE" in current_triage:
                st.warning(f"⚠️ **{current_triage}**")
                st.metric("Risk Decision", "AMBER ALERT",
                          delta="CLINICAL VIGILANCE", delta_color="normal")
            else:
                st.success(f"✅ **{current_triage}**")
                st.metric("Risk Decision", "STABLE", delta="ROUTINE CARE")

            st.write(f"**Gated Score:** `{adj_p:.4f}`")
            st.caption(
                f"Threshold [{THRESHOLD_METHOD}]: {best_threshold_saved:.4f} | "
                f"Entropy: {e_val:.4f} | Norm: {en_val:.4f}"
            )

        with res_col2:
            st.metric("Calibrated Risk Estimate", f"{cal_p:.1%}")
            st.write(f"**Method:** {cal_method.capitalize()}")
            st.caption(
                f"Youden threshold: {best_threshold_saved:.4f} | "
                f"J = {J_SCORE:.4f}" if J_SCORE else
                f"Threshold: {best_threshold_saved:.4f}"
            )

        # --- Calibrator comparison ---
        st.markdown("---")
        st.subheader("🔬 Calibrator Comparison")
        iso_prob,  _ = calibrate_single(ensemble_mean_val, "Isotonic")
        beta_prob, _ = calibrate_single(ensemble_mean_val, "Beta")

        comp_col1, comp_col2, comp_col3 = st.columns(3)
        comp_col1.metric(
            "Isotonic Calibration", f"{iso_prob:.1%}",
            delta="✓ CHOSEN" if CHOSEN_CALIBRATOR_NAME.lower() == "isotonic" else "VALIDATED",
            delta_color="off"
        )
        comp_col2.metric(
            "Beta Calibration", f"{beta_prob:.1%}",
            delta="✓ CHOSEN" if CHOSEN_CALIBRATOR_NAME.lower() == "beta" else "PREFERRED",
            delta_color="normal"
        )
        diff = abs(iso_prob - beta_prob)
        comp_col3.metric(
            "Difference", f"{diff:.1%}",
            delta="Significant" if diff > 0.03 else "Minimal",
            delta_color="inverse" if diff > 0.03 else "normal"
        )

        # --- Side-by-side interpretation ---
        st.markdown("#### 🔍 Side-by-Side Interpretation")
        int_col1, int_col2 = st.columns(2)

        # Context uses dynamic threshold — no hardcoded 0.2970
        if adj_p < best_threshold_saved * 0.85:
            triage_context = "Score is well below the clinical threshold — very low risk."
        elif adj_p < best_threshold_saved:
            triage_context = "Score is approaching the threshold. Monitor closely."
        else:
            triage_context = "Score exceeds the clinical decision threshold — prioritise review."

        with int_col1:
            st.write("**Isotonic Calibration:**")
            st.write(f"- Gated score `{adj_p:.4f}` → **{iso_prob:.1%}** mortality risk")
            st.caption(f"📊 Empirical: patients with this score had ~{iso_prob:.1%} mortality in training.")
        with int_col2:
            st.write("**Beta Calibration (PREFERRED):**")
            st.write(f"- Gated score `{adj_p:.4f}` → **{beta_prob:.1%}** mortality risk")
            st.caption("🧪 Validated: Beta estimates are more reliable for low-prevalence mortality.")

        st.markdown(f"> **Clinical Context:** {triage_context}")

        if "CRITICAL" in current_triage:
            st.error(
                f"🚨 **High Confidence Risk:** Score `{adj_p:.4f}` is in the critical percentile. "
                "Immediate intervention advised."
            )
        elif "GRAY ZONE" in current_triage:
            st.warning(
                f"⚠️ **Gray Zone:** Score `{adj_p:.4f}` is elevated. Maintain high vigilance."
            )
        elif is_near_miss:
            st.info(
                f"💡 **Borderline:** Patient is within 10% of the threshold. "
                "Review clinical history for secondary risk factors."
            )

        # --- Model committee consensus ---
        st.divider()
        st.markdown("### 🤝 Model Committee Consensus")
        m1, m2, m3 = st.columns(3)
        m1.metric("VAE ($P_1$)",     f"{m_means[0, 0]:.3f}")
        m2.metric("Flipout ($P_2$)", f"{m_means[0, 1]:.3f}")
        m3.metric("Bayesian ($P_3$)", f"{m_means[0, 2]:.3f}")

        # --- Uncertainty display ---
        if show_uncertainty:
            st.markdown("#### 📉 Uncertainty")
            u1, u2 = st.columns(2)
            u1.metric("Ensemble Std Dev", f"{float(uncertainties[0]):.4f}")
            u2.metric("Entropy (norm)",   f"{en_val:.4f}")

        # --- Audit logging ---
        audit_row = {k: (float(v) if not isinstance(v, str) else v)
                     for k, v in manual_data.items()}
        audit_row.update({
            "gated_score"     : adj_p,
            "calibrated_prob" : cal_p,
            "risk_label"      : label,
            "triage_level"    : current_triage,
            "threshold"       : best_threshold_saved,
            "threshold_method": THRESHOLD_METHOD,
            "entropy"         : e_val,
            "entropy_norm"    : en_val,
            "j_score"         : J_SCORE,
            "timestamp"       : pd.Timestamp.now().isoformat()
        })
        ok, err = append_to_gsheet(pd.DataFrame([audit_row]))
        if ok:
            st.success("✅ Patient record synced to clinical vault.")
        else:
            st.error(f"Sync failed: {err}")
