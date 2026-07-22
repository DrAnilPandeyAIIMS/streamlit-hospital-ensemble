import gc
import sys
import subprocess

# Force-install huggingface_hub if missing (required for bayesian model download)
try:
    import huggingface_hub
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "huggingface_hub", "--quiet"], check=True)

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
    roc_auc_score,
    roc_curve
)
from sklearn.calibration import calibration_curve  
import gc
tf.keras.backend.clear_session()
gc.collect()
# Force TF to use minimum memory
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
# Limit TF memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
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
This system employs a memory-optimized four-model Bayesian ensemble to provide
real-time mortality risk assessment. All risk thresholds are derived mathematically via
Youden's J statistic, calibrated for maximum sensitivity with minimum false alerts.
Model weights are performance-normalized (Rokach 2010): w_k = (AUC_k - 0.5) / Σ(AUC_j - 0.5).
""")
st.caption(f"System Operational | Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | App v2.0 — Smart imputation active")

with st.spinner("⏳ Initializing models..."):
    pass

# ============================================================
# PATHS
# ============================================================
BASE_DIR   = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR              = MODELS_DIR / "outputs" / "ensemble_model_1"


# ============================================================
# PREOPERATIVE-ONLY RISK ASSESSMENT — SEPARATE MODEL, SEPARATE UI
# ============================================================
# Distinct pathway from the deployed 4-model Bayesian ensemble.
# Uses ONLY the 26 preoperative features (Methods, para 92) —
# no postoperative feature is required, so it is usable before
# surgery, when postoperative data genuinely does not exist yet.
#
# Model: HistGradientBoostingClassifier, trained on the same
# 697-patient training split (random_state=27) as the main
# pipeline. Chosen specifically because it natively supports
# missing values (NaN) as a first-class input — a missing lab
# is treated as informative and routed through a learned split
# direction, rather than silently defaulted to 0.0 (which would
# be numerically nonsensical and clinically misleading).
#
# Validated performance:
#   Complete preop data:        AUC = 0.889 (held-out n=233)
#   ~30% of lab values missing: AUC = 0.835 (robustness check)
# The full postoperative ensemble (67 features, 24-48h window)
# remains at AUC=0.9586 — this pre-op tool is intentionally a
# separate, lower-stakes instrument matched to the amount of
# information genuinely available before surgery.
# ============================================================

PREOP_MODEL_PATH    = MODELS_DIR / "preop_model_hgb.joblib"
PREOP_ASA_MAP_PATH  = MODELS_DIR / "preop_asa_map.joblib"
PREOP_FEATNAMES_PATH = MODELS_DIR / "preop_feature_names.joblib"

PREOP_ASA_MAP = {
    'ASA_one': 1, 'ASA_two': 2, 'ASA_three': 3, 'ASA_four': 4, 'ASA-E': 5
}
PREOP_COMORBIDITIES = [
    'HIV+', 'def_Anemia', 'R_Arth', 'c_Pulm', 'DM', 'htn_C',
    'hypo_Thy', 'liver_D', 'Mets', 'Obesity', 'ren_Fail',
    'Tumor', 'MI', 'BA', 'CVA', 'ChroLiverDis', 'Hemiplegia'
]
PREOP_LABS = [
    'PreOpTLC', 'PreopUrea', 'PreopCreat', 'PreopSodium',
    'PreopPotassium', 'PreOpBilT', 'PreOpBilD', 'PreOpSGOT'
]
PREOP_FEATURES = PREOP_COMORBIDITIES + ['ASAclassification'] + PREOP_LABS

LAB_HINTS = {
    'PreOpTLC':       ("Pre-op Total Leukocyte Count (cells/mm\u00b3)", "typical 4,000-11,000"),
    'PreopUrea':      ("Pre-op Urea (mg/dL)",                       "typical 15-45"),
    'PreopCreat':     ("Pre-op Creatinine (mg/dL)",                 "typical 0.6-1.3"),
    'PreopSodium':    ("Pre-op Sodium (mEq/L)",                     "typical 135-145"),
    'PreopPotassium': ("Pre-op Potassium (mEq/L)",                  "typical 3.5-5.0"),
    'PreOpBilT':      ("Pre-op Total Bilirubin (mg/dL)",            "typical 0.2-1.2"),
    'PreOpBilD':      ("Pre-op Direct Bilirubin (mg/dL)",           "typical 0.0-0.3"),
    'PreOpSGOT':      ("Pre-op SGOT/AST (U/L)",                     "typical 8-40"),
}


@st.cache_resource
def load_preop_model():
    if not PREOP_MODEL_PATH.exists():
        st.error(f"\U0001F6A8 Preop model not found at {PREOP_MODEL_PATH}. "
                 "Place preop_model_hgb.joblib in the models/ directory.")
        st.stop()
    return joblib.load(PREOP_MODEL_PATH)


def _build_preop_dataframe_from_manual(inputs: dict) -> pd.DataFrame:
    row = {}
    for f in PREOP_COMORBIDITIES:
        row[f] = float(inputs.get(f, 0.0))
    row['ASAclassification'] = PREOP_ASA_MAP.get(inputs.get('ASAclassification'), np.nan)
    for f in PREOP_LABS:
        v = inputs.get(f, None)
        row[f] = np.nan if v is None else float(v)
    return pd.DataFrame([row])[PREOP_FEATURES]


def _build_preop_dataframe_from_csv(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df.columns = df.columns.str.strip().str.strip('"')

    if 'ASAclassification' in df.columns:
        df['ASAclassification'] = df['ASAclassification'].astype(str).str.strip().map(PREOP_ASA_MAP)
    else:
        df['ASAclassification'] = np.nan

    for f in PREOP_LABS:
        if f in df.columns:
            df[f] = pd.to_numeric(df[f], errors='coerce')
        else:
            df[f] = np.nan

    for f in PREOP_COMORBIDITIES:
        if f in df.columns:
            df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0.0)
        else:
            df[f] = 0.0

    return df[PREOP_FEATURES]


def render_preop_missing_summary(df_aligned: pd.DataFrame):
    missing_counts = df_aligned[PREOP_LABS + ['ASAclassification']].isna().sum()
    missing_counts = missing_counts[missing_counts > 0]
    if len(missing_counts) > 0:
        st.warning(
            "\u26A0\uFE0F The following fields have missing values and will be handled "
            "by the model's native missing-value logic (not imputed with a "
            "guessed number):\n\n" +
            "\n".join(f"- **{col}**: {n} patient(s) missing" for col, n in missing_counts.items())
        )
    else:
        st.success("\u2705 No missing preoperative fields detected.")


def run_preop_assessment_ui():
    st.header("\U0001FA7A Preoperative Risk Assessment (Separate Model)")
    st.info(
        "This is a **distinct, lower-stakes model** from the main postoperative "
        "ensemble. It uses only the **26 preoperative features** (comorbidities, "
        "ASA classification, preoperative labs) - nothing that requires surgery "
        "to have already happened. Validated AUC = 0.889 on complete data; "
        "0.835 with ~30%% of lab values missing. This is NOT a substitute for "
        "the full 67-feature postoperative ensemble (AUC=0.9586), which remains "
        "the tool intended for the 24-48h postoperative monitoring window."
    )

    model = load_preop_model()
    entry_mode = st.radio("Preop entry mode", ["Manual Entry", "Batch CSV"], key="preop_entry_mode")

    if entry_mode == "Manual Entry":
        with st.form("preop_entry_form"):
            st.subheader("Comorbidities")
            inputs = {}
            cols = st.columns(3)
            for i, f in enumerate(PREOP_COMORBIDITIES):
                with cols[i % 3]:
                    inputs[f] = 1.0 if st.checkbox(f, value=False, key=f"preop_{f}") else 0.0

            st.subheader("ASA Classification")
            inputs['ASAclassification'] = st.selectbox(
                "ASA Classification", list(PREOP_ASA_MAP.keys()), key="preop_asa")

            st.subheader("Preoperative Labs - leave blank if not available")
            lab_cols = st.columns(2)
            for i, f in enumerate(PREOP_LABS):
                label, hint = LAB_HINTS[f]
                with lab_cols[i % 2]:
                    raw_val = st.text_input(f"{label} ({hint})", value="", key=f"preop_lab_{f}")
                    inputs[f] = float(raw_val) if raw_val.strip() != "" else None

            submitted = st.form_submit_button("Get Preoperative Risk Estimate")

        if submitted:
            df_aligned = _build_preop_dataframe_from_manual(inputs)
            render_preop_missing_summary(df_aligned)

            proba = model.predict_proba(df_aligned)[:, 1][0]
            st.divider()
            st.metric("Preoperative Mortality Risk Estimate", f"{proba:.1%}")
            if proba >= 0.5:
                st.error("Elevated preoperative risk signal - consider further preoperative optimisation / anaesthetic risk discussion.")
            elif proba >= 0.2:
                st.warning("Moderate preoperative risk signal.")
            else:
                st.success("Lower preoperative risk signal (based on available preoperative data only).")
            st.caption(
                "This estimate reflects only preoperative information. It does not "
                "account for intraoperative events or postoperative course, and "
                "should be interpreted as a preoperative planning aid, not a final "
                "postoperative risk assessment."
            )

    else:  # Batch CSV
        uploaded = st.file_uploader(
            "Upload preoperative patient records (CSV). Leave lab cells blank "
            "if not yet available - do not enter 0.",
            type=["csv"], key="preop_csv_upload")
        if uploaded:
            df_raw = pd.read_csv(uploaded)
            df_aligned = _build_preop_dataframe_from_csv(df_raw)
            render_preop_missing_summary(df_aligned)

            probas = model.predict_proba(df_aligned)[:, 1]
            results = df_raw.copy()
            results['Preop_Risk_Estimate'] = np.round(probas, 4)
            results['Preop_Risk_Band'] = np.select(
                [probas >= 0.5, probas >= 0.2],
                ['Elevated', 'Moderate'],
                default='Lower'
            )
            st.divider()
            st.dataframe(results, use_container_width=True)
            st.download_button(
                "Download results as CSV",
                results.to_csv(index=False).encode(),
                "preop_risk_results.csv", "text/csv")


RAW_THRESHOLD_PATH       = OUTPUTS_DIR / "threshold_raw.json"
BETA_CALIBRATOR_PATH     = OUTPUTS_DIR / "beta_reg.pkl"
ISOTONIC_CALIBRATOR_PATH = OUTPUTS_DIR / "isotonic_reg.pkl"
PLATT_CALIBRATOR_PATH    = OUTPUTS_DIR / "platt_calibrator.pkl"
CALIBRATED_INFO_PATH     = OUTPUTS_DIR / "chosen_calibrator_info.json"
PERCENTILE_INFO_PATH     = OUTPUTS_DIR / "percentile_info.json"

DEBUG   = os.getenv("DEBUG", "false").lower() == "true"
EPS     = 1e-9
MC_RUNS = int(os.getenv("MC_RUNS", "30"))
IS_CLOUD = os.getenv("STREAMLIT_SERVER_HEADLESS", "false") == "true"

# ── Validate cached model files ───────────────────────────────
def _safe_delete(p):
    try:
        if Path(p).exists():
            Path(p).unlink()
    except Exception:
        pass

_models_dir = BASE_DIR / "models"

# ── CHANGE 1: model_2 now v2 — 73.5 MB ───────────────────────
# Delete old model_2_probabilistic.h5 if present (wrong β)
_m2_old = _models_dir / "model_2_probabilistic.h5"
if _m2_old.exists():
    _safe_delete(_m2_old)

# model_2_probabilistic_v2.h5 must be ~73.5 MB
_m2 = _models_dir / "model_2_probabilistic_v2.h5"
if _m2.exists() and _m2.stat().st_size < 50_000_000:
    _safe_delete(_m2)

# model_1_custom.h5 — correct file is 35.6 MB
_m1 = _models_dir / "model_1_custom.h5"
if _m1.exists() and _m1.stat().st_size < 10_000_000:
    _safe_delete(_m1)

# bayesian_model/ — only delete if incomplete/corrupted, not on every rerun.
# (Previously this ran unconditionally on every script execution, which on
# Streamlit's rerun model caused concurrent executions to race: one thread
# downloading/extracting while another deleted the folder mid-operation,
# producing intermittent FileNotFoundError on both the model variables and
# the downloaded zip file.)
_bay_folder = _models_dir / "bayesian_model"
_bay_pb = _bay_folder / "saved_model.pb"
_bay_vars = _bay_folder / "variables" / "variables.index"
if _bay_folder.exists() and not (_bay_pb.exists() and _bay_vars.exists()):
    import shutil
    shutil.rmtree(str(_bay_folder), ignore_errors=True)
    print("Removed incomplete bayesian_model folder for re-download")

# vae_model.h5 must be ~0.8 MB
_vae = _models_dir / "vae_model.h5"
if _vae.exists() and _vae.stat().st_size < 100_000:
    _safe_delete(_vae)

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
# 3. LOAD THRESHOLD JSON
# ============================================================
_CLOUD_FALLBACK = {
    "best_threshold"      : 0.4001,
    "best_t_raw"          : 0.6986,
    "threshold_method"    : "fallback",
    "high_risk_threshold" : 0.6250,
    "gamma"               : 0.10,
    "gamma_safety"        : 0.10,
    "k_steepness"         : 5.5794,
    "power_ramp"          : 1.10,
    "suppression_mult"    : 0.0893,
    "score_floor"         : 0.2678,
    "prevalence"          : 0.05594,
    "weight_vae"          : 0.2587,
    "weight_m1"           : 0.2337,
    "weight_m2"           : 0.2679,
    "weight_bay"          : 0.2398,
    "vae_weight"          : 0.2587,
    "m2_weight"           : 0.2679,
    "majority_votes"      : 3,
    "n_models"            : 4,
    "consensus_threshold" : 0.58,
    "vote_threshold_vae"  : 0.6259,
    "vote_threshold_mid"  : 0.4830,
    "vote_threshold_m2"   : 0.6086,
    "vote_threshold_bay"  : 0.6077,
    "vae_gate_threshold"  : 0.3130,
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
# 4. UNPACK ALL THRESHOLD KEYS
# ============================================================
best_threshold_saved = thr_data.get("best_threshold",      _CLOUD_FALLBACK["best_threshold"])
THRESHOLD_METHOD     = thr_data.get("threshold_method",    _CLOUD_FALLBACK["threshold_method"])
HIGH_RISK_BOUNDARY   = thr_data.get("high_risk_threshold", _CLOUD_FALLBACK["high_risk_threshold"])
GAMMA            = thr_data.get("gamma", thr_data.get("gamma_safety", 0.10))
K_STEEPNESS      = thr_data.get("k_steepness",         _CLOUD_FALLBACK["k_steepness"])
POWER_RAMP       = thr_data.get("power_ramp",          _CLOUD_FALLBACK["power_ramp"])
SUPPRESSION_MULT = thr_data.get("suppression_mult",    _CLOUD_FALLBACK["suppression_mult"])
SCORE_FLOOR      = thr_data.get("score_floor",         _CLOUD_FALLBACK["score_floor"])
PREVALENCE       = thr_data.get("prevalence",          _CLOUD_FALLBACK["prevalence"])
VAE_WEIGHT       = thr_data.get("weight_vae",  thr_data.get("vae_weight", 0.2587))
M1_WEIGHT        = thr_data.get("weight_m1",   0.2337)
M2_WEIGHT        = thr_data.get("weight_m2",   thr_data.get("m2_weight",  0.2679))
BAY_WEIGHT       = thr_data.get("weight_bay",  0.2398)
MAJORITY_VOTES   = thr_data.get("majority_votes",      _CLOUD_FALLBACK["majority_votes"])
CAUTION_WEIGHT   = thr_data.get("caution_weight",      _CLOUD_FALLBACK["caution_weight"])
ENTROPY_MIN      = thr_data.get("entropy_min",         _CLOUD_FALLBACK["entropy_min"])
ENTROPY_MAX      = thr_data.get("entropy_max",         _CLOUD_FALLBACK["entropy_max"])

THR_SEARCH_DIAG  = thr_data.get("thr_search_diag")
THR_ENTROPY_DIAG = thr_data.get("thr_entropy_diag")
FP_SEARCH_DIAG   = thr_data.get("fp_search_diag")
FP_ENTROPY_DIAG  = thr_data.get("fp_entropy_diag")
J_SCORE          = thr_data.get("j_score")

SAVED_RECALL     = thr_data.get("recall")
SAVED_TP         = thr_data.get("true_positives")
SAVED_FP         = thr_data.get("false_positives")
SAVED_TN         = thr_data.get("true_negatives")
SAVED_FN         = thr_data.get("false_negatives")

if HIGH_RISK_BOUNDARY <= best_threshold_saved:
    HIGH_RISK_BOUNDARY = 1.0

# ============================================================
# 5. TRIAGE CLASSIFICATION LOGIC
# ============================================================
def triage_levels_logic(score, threshold, high_risk_boundary=None):
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
                 int(SAVED_FP) if SAVED_FP is not None else 0)]
        if THR_SEARCH_DIAG is not None:
            rows.append(("Search",  THR_SEARCH_DIAG,  int(FP_SEARCH_DIAG)  if FP_SEARCH_DIAG  is not None else 0))
        if THR_ENTROPY_DIAG is not None:
            rows.append(("Entropy", THR_ENTROPY_DIAG, int(FP_ENTROPY_DIAG) if FP_ENTROPY_DIAG is not None else 0))
        diag_df = pd.DataFrame(rows, columns=["Method", "Threshold", "FPs"])
        diag_df["FPs"] = diag_df["FPs"].astype(int)
        diag_df["Selected"] = diag_df["Method"].apply(
            lambda m: "✅" if m.startswith("Youden") else "")
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
    st.markdown("**Pipeline Parameters — 4-Model Ensemble (Performance-Normalized Weights)**")
    st.markdown(f"""
    - **Models:** VAE (w={VAE_WEIGHT:.4f}) + Flipout M1 (w={M1_WEIGHT:.4f}) + Probabilistic M2 (w={M2_WEIGHT:.4f}) + Bayesian (w={BAY_WEIGHT:.4f})
    - **Weight method:** Performance-normalized — w_k = (AUC_k − 0.5) / Σ(AUC_j − 0.5) (Rokach 2010)
    - **Gate:** Majority ({MAJORITY_VOTES}/4 votes required)
    - **Entropy range:** `{ENTROPY_MIN:.4f}` → `{ENTROPY_MAX:.4f}`
    - **Gamma (entropy blend):** `{GAMMA}`
    - **K steepness / Power ramp:** `{K_STEEPNESS}` / `{POWER_RAMP}`
    - **Suppression mult / Score floor:** `{SUPPRESSION_MULT:.4f}` / `{SCORE_FLOOR:.4f}`
    - **T_screen / HIGH_RISK:** `{best_threshold_saved:.4f}` / `{HIGH_RISK_BOUNDARY:.4f}`
    - **Ensemble AUC:** `{thr_data.get("ensemble_auc", "—")}`
    - **Youden J:** `{thr_data.get("youden_J", "—")}`
    """)

# ============================================================
# 7. ARTIFACT LOADING
# ============================================================
chosen_calibrator_info  = load_json_safe(CALIBRATED_INFO_PATH) or {}
CHOSEN_CALIBRATOR_NAME  = chosen_calibrator_info.get("chosen_calibrator", "None")

percentile_info   = load_json_safe(PERCENTILE_INFO_PATH, {})
IS_FROZEN         = percentile_info.get("frozen", False)
FROZEN_PERCENTILE = percentile_info.get("percentile")

feature_names_raw = load_pickle_safe(MODELS_DIR / "feature_names.pkl")
feature_names     = list(feature_names_raw) if feature_names_raw is not None else []

label_encoder = None

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
SCALABLE_FEATURES = NUMERIC_FEATURES + ordinal_variables
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
    SCALABLE_INDICES = [
        feature_names.index(col)
        for col in SCALABLE_FEATURES
        if col in feature_names
    ]
    scalable_columns = [col for col in SCALABLE_FEATURES if col in feature_names]
except Exception as e:
    st.error(f"🚨 Index Mapping Error: {e}")
    st.stop()

if scaler is not None:
    if scaler.n_features_in_ != len(scalable_columns):
        st.warning(
            f"⚠️ Scaler has {scaler.n_features_in_} features, "
            f"expected {len(scalable_columns)}. "
            f"Using scaler on available {min(scaler.n_features_in_, len(scalable_columns))} features."
        )
        SCALABLE_INDICES = [
            feature_names.index(col)
            for col in SCALABLE_FEATURES[:scaler.n_features_in_]
            if col in feature_names
        ]
        scalable_columns = SCALABLE_FEATURES[:scaler.n_features_in_]

st.success(f"✅ System aligned: {len(feature_names)} features | {len(SCALABLE_INDICES)} scalable | 4-model ensemble | MC={MC_RUNS} | v2.0")

# ============================================================
# 9. CALIBRATORS
# ============================================================
class BetaCalibrator:
    def __init__(self, a, b):
        self.a = a; self.b = b
    def transform(self, p):
        p = np.clip(p, EPS, 1 - EPS)
        return (p ** self.a) / ((p ** self.a) + ((1 - p) ** self.b))

beta_calibrator  = load_pickle_safe(BETA_CALIBRATOR_PATH)
iso_calibrator   = load_pickle_safe(ISOTONIC_CALIBRATOR_PATH)
platt_calibrator = load_pickle_safe(PLATT_CALIBRATOR_PATH)

# ============================================================
# 10. CALIBRATION METRICS
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
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Fraction of positives")
    ax.set_title(title); ax.legend()
    st.pyplot(fig)

# ============================================================
# 11. MODEL REGISTRY & MANAGER
# ============================================================
MODEL_FILES = {
    "vae_model": {
        "id"  : "17do6Clm4WH_Us2Y_FVsANjj7_ktBQvlO",
        "path": MODELS_DIR / "vae_model.h5"
    },
    "model_1": {
        "id"  : "1cGQUH6cM_18rmJ0_2RuNcdt4DWFSpxCG",
        "path": MODELS_DIR / "model_1_custom.h5",
    },
    "model_2": {
        "id"  : "1q6CVPm_WwIxLq_2tDCEmNs_Mg0Xt2kdq",
        "path": MODELS_DIR / "model_2_probabilistic_v2.h5",
    },
    "bayesian_model": {
        "id"  : "1tERLaaB5E8A3DfqFUo9YMdMtIvrCuGk1",
        "path": MODELS_DIR / "bayesian_model",
        "zip" : True,
        "hf_repo": "akpandet/perioperative-models",
        "hf_file": "bayesian_model_corrected.zip",
        "files": {}
    }
}


def _model_is_cached(key: str) -> bool:
    info   = MODEL_FILES[key]
    path   = Path(info["path"])
    is_zip = info.get("zip", False)
    if is_zip:
        vars_dir   = path / "variables"
        data_files = list(vars_dir.glob("*.data*")) if vars_dir.exists() else []
        pb_exists  = (path / "saved_model.pb").exists() if path.exists() else False
        result = (path.exists() and path.is_dir() and pb_exists and
                  vars_dir.exists() and len(data_files) > 0)
        print(f"_model_is_cached({key}): folder={path.exists()} pb={pb_exists} vars={vars_dir.exists()} data={len(data_files)} result={result}")
        return result
    else:
        return path.exists() and path.stat().st_size > 100_000


# ============================================================
# ── FIXED _ensure_model_downloaded ───────────────────────────
# CHANGE: fuzzy=True on ALL gdown attempts (both Attempt 1 and
#         Attempt 2).  fuzzy=True makes gdown handle Google's
#         virus-scan confirmation page automatically, which is
#         the root cause of the "0 bytes / cannot retrieve"
#         error on large files (>100 MB).
#
# Three-layer fallback:
#   1. gdown fuzzy=True  (handles scan page automatically)
#   2. gdown confirm URL + fuzzy=True  (explicit confirm token)
#   3. requests streaming  (manual cookie-based confirm)
# ============================================================
def _ensure_model_downloaded(model_key: str) -> Path:
    import gdown
    import requests
    import zipfile
    info     = MODEL_FILES[model_key]
    path     = Path(info["path"])
    drive_id = info["id"]
    is_zip   = info.get("zip", False)

    # ── Already present? ─────────────────────────────────────
    if is_zip:
        if path.exists() and path.is_dir() and (path / "saved_model.pb").exists():
            return path
    else:
        if path.exists() and path.stat().st_size > 100_000:
            return path

    # ── Clean stale partial downloads ────────────────────────
    if is_zip:
        if path.exists() and path.is_dir():
            try:
                import shutil
                shutil.rmtree(str(path))
            except Exception:
                pass
    else:
        if path.exists():
            try:
                path.unlink()
            except Exception:
                pass

    path.parent.mkdir(parents=True, exist_ok=True)
    dl_path = str(path.parent / f"{model_key}.zip") if is_zip else str(path)

    # ── ATTEMPT 0: Hugging Face Hub (reliable, no rate limits) ─
    # Used for bayesian_model (132 MB) — Google Drive blocks large
    # files from cloud IPs. HF has no such restriction.
    hf_repo = info.get("hf_repo")
    hf_file = info.get("hf_file")
    if hf_repo and hf_file and is_zip:
        import requests, zipfile
        # Direct HF URL — works for public repos, no hf_hub_download needed
        hf_url = f"https://huggingface.co/{hf_repo}/resolve/main/{hf_file}?download=true"
        print(f"📥 Downloading {model_key} from HF direct URL...")
        print(f"  URL: {hf_url}")
        # Unique per-process filename — prevents concurrent Streamlit
        # reruns from colliding on the same zip path (one execution's
        # unlink() succeeding while another still expects the file,
        # which produced the "bayesian_model_corrected.zip not found"
        # error seen when multiple reruns overlapped).
        import os as _os
        zip_dest = path.parent / f"{model_key}_{_os.getpid()}_{hf_file}"
        try:
            r = requests.get(hf_url, stream=True, timeout=600,
                             headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            with open(zip_dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
            size = zip_dest.stat().st_size if zip_dest.exists() else 0
            print(f"  Downloaded {size:,} bytes")
            if size > 100_000:
                print(f"  Extracting zip...")
                with zipfile.ZipFile(str(zip_dest), "r") as zf:
                    zf.extractall(str(path.parent))
                zip_dest.unlink()
                # Check if saved_model.pb is directly at expected path
                if path.exists() and (path / "saved_model.pb").exists():
                    print(f"  ✅ {model_key} ready from Hugging Face.")
                    return path
                # Zip may extract to a differently-named subfolder — find saved_model.pb
                import glob
                pb_matches = glob.glob(str(path.parent / "**/saved_model.pb"), recursive=True)
                print(f"  saved_model.pb search results: {pb_matches}")
                if pb_matches:
                    import shutil
                    found_dir = Path(pb_matches[0]).parent
                    print(f"  Found model at: {found_dir} — moving to {path}")
                    if path.exists():
                        shutil.rmtree(str(path))
                    shutil.move(str(found_dir), str(path))
                    if path.exists() and (path / "saved_model.pb").exists():
                        print(f"  ✅ {model_key} ready from Hugging Face (after rename).")
                        return path
                raise RuntimeError(f"HF zip extracted but saved_model.pb not found anywhere under {path.parent}. pb_matches={pb_matches}")
            else:
                raise RuntimeError(f"HF download too small: {size} bytes — repo may be private")
        except Exception as e:
            import streamlit as _st
            _st.error(f"🔴 HF download error for {model_key}: {type(e).__name__}: {e}")
            if zip_dest.exists():
                zip_dest.unlink()
            raise RuntimeError(f"HF download failed: {type(e).__name__}: {e}") from e

    # ── ATTEMPT 1: gdown with fuzzy=True ─────────────────────
    # fuzzy=True lets gdown parse the HTML confirmation page
    # that Google shows for large files — previously missing
    # from Attempt 1, causing silent 0-byte downloads.
    print(f"📥 Downloading {model_key} (ID: {drive_id[:8]}...) — Attempt 1 (fuzzy=True)...")
    url = f"https://drive.google.com/uc?id={drive_id}"
    try:
        gdown.download(url, dl_path, quiet=False, fuzzy=True)
    except Exception as e:
        print(f"  Attempt 1 exception: {e}")

    dl_path_obj = Path(dl_path)
    if is_zip and dl_path_obj.exists() and dl_path_obj.stat().st_size > 100_000:
        print(f"  Attempt 1 zip downloaded ({dl_path_obj.stat().st_size:,} bytes) — extracting...")
        with zipfile.ZipFile(str(dl_path_obj), "r") as zf:
            zf.extractall(str(path.parent))
        dl_path_obj.unlink()
        if path.exists() and (path / "saved_model.pb").exists():
            print(f"  ✅ {model_key} ready after Attempt 1.")
            return path
    elif not is_zip and path.exists() and path.stat().st_size > 100_000:
        print(f"  ✅ {model_key} ready after Attempt 1.")
        return path

    # ── ATTEMPT 2: explicit confirm token + fuzzy=True ───────
    # Adds confirm=t to URL AND keeps fuzzy=True so gdown
    # can still handle any remaining scan page redirect.
    print(f"📥 Downloading {model_key} — Attempt 2 (confirm=t + fuzzy=True)...")
    if Path(dl_path).exists():
        try:
            Path(dl_path).unlink()
        except Exception:
            pass
    confirm_url = f"https://drive.google.com/uc?export=download&confirm=t&id={drive_id}"
    try:
        gdown.download(confirm_url, dl_path, quiet=False, fuzzy=True)
    except Exception as e:
        print(f"  Attempt 2 exception: {e}")

    dl_path_obj2 = Path(dl_path)
    if is_zip and dl_path_obj2.exists() and dl_path_obj2.stat().st_size > 100_000:
        print(f"  Attempt 2 zip downloaded ({dl_path_obj2.stat().st_size:,} bytes) — extracting...")
        import zipfile as zf2
        with zf2.ZipFile(str(dl_path_obj2), "r") as z:
            z.extractall(str(path.parent))
        dl_path_obj2.unlink()
        if path.exists() and (path / "saved_model.pb").exists():
            print(f"  ✅ {model_key} ready after Attempt 2.")
            return path
    elif not is_zip and path.exists() and path.stat().st_size > 100_000:
        print(f"  ✅ {model_key} ready after Attempt 2.")
        return path

    # ── ATTEMPT 3: requests streaming with cookie confirm ────
    # Manual HTTP fallback — handles cases where gdown's
    # internal UA triggers an extra CAPTCHA layer.
    print(f"📥 Downloading {model_key} — Attempt 3 (requests streaming)...")
    if Path(dl_path).exists():
        try:
            Path(dl_path).unlink()
        except Exception:
            pass
    try:
        session  = requests.Session()
        response = session.get(
            f"https://drive.google.com/uc?export=download&id={drive_id}",
            stream=True, timeout=60)
        token = None
        for key_c, value in response.cookies.items():
            if key_c.startswith('download_warning'):
                token = value
                break
        if token:
            response = session.get(
                f"https://drive.google.com/uc?export=download&confirm={token}&id={drive_id}",
                stream=True, timeout=600)
        write_path = Path(dl_path) if is_zip else path
        with open(write_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)
        print(f"  Attempt 3 wrote {Path(dl_path).stat().st_size if Path(dl_path).exists() else 0:,} bytes")
    except Exception as e:
        print(f"  Attempt 3 exception: {e}")

    # ── Final extraction if zip ───────────────────────────────
    final_dl = Path(dl_path)
    if is_zip and final_dl.exists() and final_dl.stat().st_size > 100_000:
        print(f"  Attempt 3 zip downloaded — extracting...")
        import zipfile as zf3
        with zf3.ZipFile(str(final_dl), "r") as z:
            z.extractall(str(path.parent))
        final_dl.unlink()
        if path.exists() and (path / "saved_model.pb").exists():
            print(f"  ✅ {model_key} ready after Attempt 3.")
            return path

    if _model_is_cached(model_key):
        return path

    size = path.stat().st_size if path.exists() else 0
    print(f"❌ All 3 attempts failed for {model_key} (Drive ID: {drive_id}). Final size: {size:,} bytes")
    raise RuntimeError(
        f"Failed to download {model_key} (Drive ID: {drive_id}). "
        f"Size: {size:,} bytes. Check Google Drive sharing settings and rate limits.")


def _load_model_cached(key: str):
    info = MODEL_FILES[key]
    path = Path(info["path"])
    if not _model_is_cached(key):
        print(f"📥 Downloading {key} from Google Drive...")
        path = _ensure_model_downloaded(key)
        size_str = f"{path.stat().st_size/1e6:.1f} MB" if path.is_file() else "SavedModel folder"
        print(f"✅ Downloaded {key} ({size_str})")
    else:
        path = Path(MODEL_FILES[key]["path"])
    model = tf.keras.models.load_model(
        path, compile=False, custom_objects=get_custom_objects())
    print(f"✅ Loaded {key} into cache")
    return model

class SingleModelManager:
    def __init__(self):
        self.current_key = None
        self.model       = None

    def load(self, key):
        self.model       = _load_model_cached(key)
        self.current_key = key
        return self.model

    def unload(self):
        if self.model is not None:
            try:
                del self.model
                tf.keras.backend.clear_session()
                gc.collect()
            except Exception:
                pass
        self.model = None; self.current_key = None

model_manager = SingleModelManager()

# ============================================================
# PRE-DOWNLOAD ALL MODELS AT STARTUP (cached — runs once only)
# Prevents timeout during inference by downloading before user
# interaction. @st.cache_resource persists across reruns.
# ============================================================
@st.cache_resource(show_spinner="⏳ Downloading models (first run only — ~2 min)...")
def _predownload_all_models():
    results = {}
    for key in ["vae_model", "model_1", "model_2", "bayesian_model"]:
        try:
            if not _model_is_cached(key):
                print(f"🔄 Pre-downloading {key}...")
                _ensure_model_downloaded(key)
                print(f"✅ Pre-download complete: {key}")
            else:
                print(f"✅ Already cached: {key}")
            results[key] = "ok"
        except Exception as e:
            print(f"❌ Pre-download failed for {key}: {e}")
            results[key] = f"failed: {e}"
    return results

_predownload_status = _predownload_all_models()

# ============================================================
# 12. TENSORFLOW CUSTOM OBJECTS
# ============================================================
def _initialize_tensorflow_components():
    import tensorflow_probability as tfp
    tfd = tfp.distributions

    def prior(kernel_size, bias_size, dtype=None):
        n = kernel_size + bias_size
        return tf.keras.Sequential([
            tfp.layers.VariableLayer(n, dtype=dtype),
            tfp.layers.DistributionLambda(
                lambda t: tfd.MultivariateNormalDiag(
                    loc=t, scale_diag=tf.ones_like(t)))])

    def posterior(kernel_size, bias_size, dtype=None):
        n = kernel_size + bias_size
        return tf.keras.Sequential([
            tfp.layers.VariableLayer(
                tfp.layers.IndependentNormal.params_size(n), dtype=dtype),
            tfp.layers.IndependentNormal(
                n, convert_to_tensor_fn=tfd.Distribution.sample)])

    class CustomDenseVariational(tfp.layers.DenseVariational):
        def __init__(self, units, make_prior_fn, make_posterior_fn,
                     kl_weight=1.0, **kwargs):
            super().__init__(units=units, make_prior_fn=make_prior_fn,
                             make_posterior_fn=make_posterior_fn,
                             kl_weight=kl_weight, **kwargs)
            self.units = units; self.kl_weight = kl_weight
        def get_config(self):
            config = super().get_config()
            config.update({"units": self.units, "kl_weight": self.kl_weight})
            return config
        @classmethod
        def from_config(cls, config):
            config["make_prior_fn"]     = prior
            config["make_posterior_fn"] = posterior
            return cls(**config)

    class DenseFlipoutLayer(tf.keras.layers.Layer):
        def __init__(self, units, activation=None,
                     kl_weight=1.0, **kwargs):
            super().__init__(**kwargs)
            self.units      = units
            self.activation = activation
            self.kl_weight  = kl_weight
        def build(self, input_shape):
            self.dense_flipout = tfp.layers.DenseFlipout(
                units=self.units,
                activation=self.activation,
                kernel_divergence_fn=lambda q, p, _:
                    tfp.distributions.kl_divergence(q, p) * self.kl_weight,
                bias_divergence_fn=lambda q, p, _:
                    tfp.distributions.kl_divergence(q, p) * self.kl_weight)
            super().build(input_shape)
        def call(self, inputs):
            return self.dense_flipout(inputs)
        def get_config(self):
            config = super().get_config()
            config.update({
                'units'     : self.units,
                'activation': self.activation,
                'kl_weight' : self.kl_weight,
            })
            return config

    def negative_log_likelihood_bernoulli(y_true, y_pred):
        return -tf.reduce_mean(
            y_true * tf.math.log(y_pred + 1e-9) +
            (1 - y_true) * tf.math.log(1 - y_pred + 1e-9))

    def negative_log_likelihood(y_true, y_pred_dist):
        return -y_pred_dist.log_prob(y_true)

    return {
        "CustomDenseVariational"            : CustomDenseVariational,
        "DenseFlipoutLayer"                 : DenseFlipoutLayer,
        "DenseFlipout"                      : tfp.layers.DenseFlipout,
        "DistributionLambda"                : tfp.layers.DistributionLambda,
        "prior"                             : prior,
        "posterior"                         : posterior,
        "negative_log_likelihood"           : negative_log_likelihood,
        "negative_log_likelihood_bernoulli" : negative_log_likelihood_bernoulli,
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
    p = np.clip(np.ravel(probs), EPS, 1 - EPS)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))

def ensure_single_output(arr):
    a = np.asarray(arr)
    if a.size and (a.min() < 0 or a.max() > 1):
        a = 1.0 / (1.0 + np.exp(-a))
    if a.ndim == 2 and a.shape[1] == 2:
        return a[:, 1]
    if a.ndim == 2 and a.shape[1] == 1:
        return a.ravel()
    return a.ravel()

def mc_forward_pass(model, x):
    """
    Custom forward pass that decouples two behaviors a single
    `training=True/False` flag would otherwise conflate:

      - Dropout layers: run in STOCHASTIC mode (training=True),
        since that randomness is exactly what Monte Carlo
        uncertainty sampling needs.
      - BatchNormalization layers: run in FROZEN mode
        (training=True is what causes this whole issue —
        it makes BatchNorm compute its normalization statistics
        from whichever patients happen to be in the CURRENT
        batch, rather than using the fixed statistics learned
        during training. That means the same patient's predicted
        risk could shift purely because of which other patients
        were uploaded alongside them — the exact instability
        traced through today's investigation.)

    All other layers (Dense, DenseVariational, Activation, etc.)
    pass through unaffected either way.
    """
    out = x
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            out = layer(out, training=False)   # frozen population stats
        elif isinstance(layer, tf.keras.layers.Dropout):
            out = layer(out, training=True)    # stochastic, for MC sampling
        else:
            out = layer(out, training=True)
    return out

@st.cache_resource
def load_small_objects():
    if scaler is None or not hasattr(scaler, "transform"):
        raise RuntimeError("Valid scaler not found. Check scaler.pkl.")
    return scaler, feature_names

scaler_cached, feature_names_cached = load_small_objects()

# ============================================================
# 14. MC INFERENCE — 4-MODEL ENSEMBLE
# ============================================================
mc_passes = MC_RUNS
def load_models_and_mc_for_batch(X_np, n_forward_passes=30, use_frozen_batchnorm=False):
    """
    use_frozen_batchnorm=False (default): preserves existing behavior —
        model(x, training=True) for every layer, including
        BatchNormalization. Batch-composition-dependent (see
        investigation notes, July 2026). Used by Batch CSV.
    use_frozen_batchnorm=True: uses mc_forward_pass() to keep Dropout
        stochastic (for MC sampling) while forcing BatchNormalization
        to use its frozen, learned statistics — patient-independent
        of whatever else is in the same call. Used by Manual Entry,
        since a batch-of-one is the worst-case scenario for the
        original behavior.
    """
    model_keys = ["vae_model", "model_1", "model_2", "bayesian_model"]
    X_tensor   = tf.convert_to_tensor(np.asarray(X_np, dtype=np.float32))

    tf.random.set_seed(42)
    np.random.seed(42)

    all_model_mc_means = []
    for key in model_keys:
        model      = model_manager.load(key)
        mc_samples = []
        for i in range(n_forward_passes):
            tf.random.set_seed(42 + i)
            if use_frozen_batchnorm:
                raw = ensure_single_output(
                    mc_forward_pass(model, X_tensor))
            else:
                raw = ensure_single_output(
                    model(X_tensor, training=True))

            # Bayesian: outputs raw logits — sigmoid mandatory
            if key == "bayesian_model":
                raw = tf.math.sigmoid(
                    tf.constant(raw, dtype=tf.float32)).numpy()

            # M2 corrected β=8.22e-8: correct direction — NO inversion
            mc_samples.append(raw)


        all_model_mc_means.append(
            np.mean(np.vstack(mc_samples), axis=0))

        model_manager.unload()
        gc.collect()

    all_model_mc_means = np.array(all_model_mc_means)  # (4, N)
    mean_per_model     = all_model_mc_means.T           # (N, 4)

    vae_p = mean_per_model[:, 0]
    m1_p  = mean_per_model[:, 1]
    m2_p  = mean_per_model[:, 2]
    bay_p = mean_per_model[:, 3]

    _w_vae = thr_data.get("weight_vae", thr_data.get("vae_weight", 0.2587))
    _w_m1  = thr_data.get("weight_m1",  0.2337)
    _w_m2  = thr_data.get("weight_m2",  thr_data.get("m2_weight", 0.2679))
    _w_bay = thr_data.get("weight_bay", 0.2398)

    base_risk = (vae_p * _w_vae +
                 m1_p  * _w_m1  +
                 m2_p  * _w_m2  +
                 bay_p * _w_bay)

    v_vae = (vae_p > thr_data.get("vote_threshold_vae", 0.6259)).astype(int)
    v_m1  = (m1_p  > thr_data.get("vote_threshold_mid", 0.4830)).astype(int)
    v_m2  = (m2_p  > thr_data.get("vote_threshold_m2",  0.6086)).astype(int)
    v_bay = (bay_p > thr_data.get("vote_threshold_bay", 0.6077)).astype(int)
    total_votes = v_vae + v_m1 + v_m2 + v_bay

    vae_mask       = (vae_p > thr_data.get("vae_gate_threshold", 0.3130))
    _cons_thr      = thr_data.get("consensus_threshold", 0.58)
    consensus_mask = ((m1_p > _cons_thr) & (m2_p > _cons_thr) & (bay_p > _cons_thr))
    _majority      = thr_data.get("majority_votes", 3)
    is_valid       = (vae_mask | consensus_mask) | (total_votes >= _majority)

    _supp          = thr_data.get("suppression_mult", SUPPRESSION_MULT)
    weighted_probs = np.where(is_valid, base_risk, base_risk * _supp)

    _floor     = thr_data.get("score_floor", SCORE_FLOOR)
    any_signal = ((vae_p > 0.05) | (m1_p > 0.05) |
                  (m2_p  > 0.05) | (bay_p > 0.05))
    weighted_probs = np.where(
        any_signal, np.maximum(weighted_probs, _floor), weighted_probs)

    avg_p       = (vae_p + m1_p + m2_p + bay_p) / 4.0
    p_clip      = np.clip(avg_p, EPS, 1 - EPS)
    entropy_raw = -(p_clip * np.log2(p_clip) +
                    (1 - p_clip) * np.log2(1 - p_clip))

    e_min = entropy_raw.min(); e_max = entropy_raw.max()
    entropy_norm = (entropy_raw - e_min) / (e_max - e_min + EPS)

    _gamma         = thr_data.get("gamma", thr_data.get("gamma_safety", 0.10))
    adjusted_probs = weighted_probs + (_gamma * entropy_norm)

    # Use Youden threshold (0.4001) — gave TP=13, FP=0, FN=0 on test set
    # best_t_raw (0.6986) is raw-space conversion — too strict for deployment
    runtime_threshold = best_threshold_saved   # 0.4001
    high_risk_thr     = thr_data.get("high_risk_threshold", HIGH_RISK_BOUNDARY)  # 0.6250

    triage_levels = [
        triage_levels_logic(score, runtime_threshold, high_risk_thr)
        for score in adjusted_probs
    ]

    ensemble_std = np.std(all_model_mc_means, axis=0)
    return (mean_per_model, adjusted_probs, ensemble_std,
            entropy_raw, entropy_norm, triage_levels)

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
            return gspread.authorize(creds), None
        except Exception:
            pass
    info = st.secrets.get("gcp_service_account")
    if not info:
        return None, "No credentials found"
    try:
        if isinstance(info, str):
            info = json.loads(info)
        creds  = Credentials.from_service_account_info(info, scopes=scopes)
        return gspread.authorize(creds), None
    except Exception as e:
        return None, f"Error: {e}"

# ============================================================
# SESSION STATE DEFAULTS
# ============================================================
if "df_input"     not in st.session_state: st.session_state["df_input"]     = None
if "last_results" not in st.session_state: st.session_state["last_results"] = None

show_confusion       = True
show_uncertainty     = True
show_raw_model_probs = True

# ============================================================
# 17. GOOGLE SHEETS: APPEND
# ============================================================
def append_to_gsheet(df, sheet_key=None, worksheet_name=None):
    if not isinstance(df, pd.DataFrame):
        try: df = pd.DataFrame(df)
        except Exception as e: return False, f"Conversion error: {e}"
    if df.empty: return True, "DataFrame is empty."
    client, err = get_gs_client_from_secrets()
    if client is None: return False, f"Auth error: {err}"
    sheet_key      = sheet_key      or st.secrets.get("gsheet_key")
    worksheet_name = worksheet_name or st.secrets.get("gsheet_worksheet", "streamlit_project Data")
    if not sheet_key: return False, "No gsheet_key found."
    if "docs.google.com" in sheet_key:
        sheet_key = sheet_key.split("/d/")[1].split("/")[0]
    try:
        sh = client.open_by_key(sheet_key)
        try: ws = sh.worksheet(worksheet_name)
        except gspread.WorksheetNotFound:
            ws = sh.add_worksheet(title=worksheet_name, rows="1000", cols="20")
            ws.append_row(list(df.columns))
        existing_headers = ws.row_values(1)
        if not existing_headers:
            existing_headers = list(df.columns)
            ws.insert_row(existing_headers, index=1)
        else:
            # Auto-extend header row for any genuinely new columns
            # (e.g. m2_prob) instead of silently dropping them via
            # reindex — this was the root cause of individual model
            # scores never reaching the vault previously.
            new_cols = [c for c in df.columns if c not in existing_headers]
            if new_cols:
                existing_headers = existing_headers + new_cols
                ws.update('A1', [existing_headers])
        df_aligned     = df.reindex(columns=existing_headers).fillna("")
        rows_to_append = df_aligned.astype(str).values.tolist()
        ws.append_rows(rows_to_append, value_input_option="USER_ENTERED")
        return True, f"Synced {len(df)} patients."
    except Exception as e:
        return False, f"Sync failed: {e}"

def log_clinical_inference(input_df, raw_p, adj_p, entropy, risk_label):
    log_entry = input_df.copy()
    log_entry["Inference_Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry["Ensemble_Raw_Prob"]   = raw_p
    log_entry["Model_Threshold"]     = best_threshold_saved
    log_entry["Threshold_Method"]    = THRESHOLD_METHOD
    log_entry["Entropy_NaturalLog"]  = entropy
    log_entry["Gated_Prob_Gamma"]    = adj_p
    log_entry["Clinical_Risk_Label"] = risk_label
    log_entry["Gamma_Penalty_Applied"] = GAMMA
    log_entry["J_Score"]             = J_SCORE
    return append_to_gsheet(log_entry)

def read_from_gsheet(n=5):
    client, err = get_gs_client_from_secrets()
    if client is None: return None, f"Error: {err}"
    sheet_key      = st.secrets.get("gsheet_key")
    worksheet_name = st.secrets.get("gsheet_worksheet", "streamlit_project Data")
    if not sheet_key: return None, "No gsheet_key"
    try:
        sh         = client.open_by_key(sheet_key)
        ws         = sh.worksheet(worksheet_name)
        all_values = ws.get_all_values()
        if not all_values or len(all_values) <= 1: return pd.DataFrame(), None
        df = pd.DataFrame(all_values[1:], columns=all_values[0])
        return df.tail(n), None
    except Exception as e:
        return None, str(e)

# ============================================================
# 18. SIDEBAR
# ============================================================
with st.sidebar:
    st.title("⚙ Model Controls")

    sidebar_view = st.radio(
        "Select Panel",
        ["🔍 Calibration (Audit)", "📊 Evaluation / Prediction"],
        index=0
    )
    st.write(f"🔹 **High-risk percentile (frozen): Top {FROZEN_PERCENTILE}%**")

    if sidebar_view == "🔍 Calibration (Audit)":
        st.subheader("🔒 Frozen Calibration State")
        st.write("**Chosen calibrator:**", CHOSEN_CALIBRATOR_NAME)
        st.write("**Threshold (Youden's J):**", f"{best_threshold_saved:.4f}")
        st.caption(f"Method: {THRESHOLD_METHOD} | J = {J_SCORE:.4f}" if J_SCORE else
                   f"Method: {THRESHOLD_METHOD}")
        st.markdown("#### 📁 Calibration Metadata")
        st.json(chosen_calibrator_info)
        if THR_SEARCH_DIAG is not None or THR_ENTROPY_DIAG is not None:
            st.markdown("#### 📊 Threshold Method Comparison")
            rows = [("Youden (primary ✅)", best_threshold_saved,
                     SAVED_FP if SAVED_FP is not None else "—")]
            if THR_SEARCH_DIAG is not None:
                rows.append(("Search  (diag)", THR_SEARCH_DIAG, FP_SEARCH_DIAG or "—"))
            if THR_ENTROPY_DIAG is not None:
                rows.append(("Entropy (diag)", THR_ENTROPY_DIAG, FP_ENTROPY_DIAG or "—"))
            _fps_df = pd.DataFrame(rows, columns=["Method", "Threshold", "FPs"])
            _fps_df["FPs"] = _fps_df["FPs"].apply(lambda x: int(x) if x != "—" else 0)
            st.dataframe(_fps_df, hide_index=True, use_container_width=True)
        if SAVED_RECALL is not None:
            st.markdown("#### 📈 Training Performance")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Recall", f"{SAVED_RECALL:.1%}")
            m2.metric("TP", SAVED_TP or "—")
            m3.metric("FP", SAVED_FP or "—")
            m4.metric("FN", SAVED_FN or "—")
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
                if err: st.error(f"Vault Connection Error: {err}")
                elif df_history is not None and not df_history.empty:
                    st.dataframe(df_history, use_container_width=True)
                else: st.info("Clinical Vault is empty.")
        st.markdown("---")
        st.caption(f"🔒 4-Model Ensemble | T_screen={best_threshold_saved:.4f} | AUC={thr_data.get('ensemble_auc','—')}")

    elif sidebar_view == "📊 Evaluation / Prediction":
        st.subheader("Prediction Configuration")
        st.selectbox("Calibration mode (display only)",
                     ["Auto (chosen)", "Beta", "Isotonic", "None"],
                     index=0, key="calib_mode")
        st.markdown("---")
        st.subheader("🩺 Clinical Risk Stratification")
        stratification_mode = st.radio(
            "Risk stratification method",
            ["Percentile-based (Research Only)", "Safety-First Threshold (Clinical)"],
            index=1, key="strat_method_radio")
        is_percentile_mode = stratification_mode.startswith("Percentile")
        if is_percentile_mode:
            if not IS_FROZEN or FROZEN_PERCENTILE is None:
                st.error("❌ Frozen percentile cutoff missing."); st.stop()
            st.write(f"🔹 **Relative Cutoff: Top {FROZEN_PERCENTILE}%**")
        else:
            st.write(f"🔹 **Youden Threshold: {best_threshold_saved:.4f}**")
        st.markdown("---")
        st.subheader("Metrics Display")
        show_uncertainty     = st.checkbox("Show Uncertainty (Std & Entropy)", value=True)
        show_raw_model_probs = st.checkbox("Show Raw Model Probabilities", value=True)
        show_confusion       = st.checkbox("Show Confusion Matrix / Scores", value=True)

# ============================================================
# 19. PREPROCESSING
# ============================================================
ASA_ORDINAL_MAP = {
    'ASA_one':   1,
    'ASA_two':   2,
    'ASA_three': 3,
    'ASA_four':  4,
    'ASA-E':     5
}
def apply_asa_encoding(df_col):
    return df_col.map(ASA_ORDINAL_MAP).fillna(1).astype(float)

def apply_training_scaling(df: pd.DataFrame) -> np.ndarray:
    try:
        working_df = df.copy()
        if 'ASAclassification' in working_df.columns:
            if working_df['ASAclassification'].dtype == object:
                valid_labels = set(ASA_ORDINAL_MAP.keys())
                working_df['ASAclassification'] = working_df['ASAclassification'].apply(
                    lambda x: x if x in valid_labels else 'ASA_one')
                working_df['ASAclassification'] = apply_asa_encoding(
                    working_df['ASAclassification'])
        df_aligned = working_df.reindex(columns=feature_names, fill_value=0.0).fillna(0.0)
        X = df_aligned.values.astype(np.float32)
        if scaler is not None and len(SCALABLE_INDICES) > 0:
            X[:, SCALABLE_INDICES] = scaler.transform(X[:, SCALABLE_INDICES])
        return X
    except Exception as e:
        st.error(f"⚠️ Preprocessing FAILED: {e}"); st.stop()

# ============================================================
# 20. CALIBRATION WRAPPERS
# ============================================================
def calibrate_probs(arr, mode="Auto (chosen)",
                    chosen_calibrator_info=None, beta_calibrator=None,
                    iso_calibrator=None, platt_calibrator=None):
    _info  = chosen_calibrator_info or globals().get("chosen_calibrator_info", {})
    _beta  = beta_calibrator  or globals().get("beta_calibrator")
    _iso   = iso_calibrator   or globals().get("iso_calibrator")
    _platt = platt_calibrator or globals().get("platt_calibrator")
    arr       = np.asarray(arr, dtype=float)
    p_clipped = np.clip(arr, EPS, 1 - EPS)
    chosen = mode
    if mode == "Auto (chosen)":
        chosen = (_info or {}).get("chosen_calibrator", "platt")
    chosen_lower = chosen.lower()
    if chosen_lower == "platt":
        if _platt is not None:
            try: return np.clip(_platt.predict_proba(p_clipped.reshape(-1,1))[:,1], EPS, 1-EPS)
            except Exception: pass
        return p_clipped
    if chosen_lower == "isotonic":
        if _iso is not None:
            try: return np.clip(_iso.predict(p_clipped), EPS, 1-EPS)
            except Exception: pass
        return p_clipped
    if chosen_lower == "beta":
        if _beta is not None:
            try:
                if hasattr(_beta, "a") and hasattr(_beta, "b"):
                    return np.clip(_beta.transform(p_clipped), EPS, 1-EPS)
                return np.clip(_beta.predict_proba(p_clipped.reshape(-1,1))[:,1], EPS, 1-EPS)
            except Exception: pass
        return p_clipped
    return p_clipped

def calibrate_single(val: float, mode: str):
    res_arr = calibrate_probs(np.array([val]), mode=mode)
    chosen  = mode
    if mode == "Auto (chosen)":
        chosen = (chosen_calibrator_info or {}).get("chosen_calibrator", "uncalibrated")
    return float(res_arr[0]), chosen

def calibrate_probs_runtime(arr):
    mode = st.session_state.get("calib_mode", "Auto (chosen)")
    return calibrate_probs(arr, mode)

# ============================================================
# 21. MAIN UI
# ============================================================
results = None
mode    = st.radio("Select Entry Mode", ["Batch CSV", "Manual Entry", "Preoperative Assessment"],
                   key="entry_mode_selector")

# ============================================================
# 21a. BATCH CSV
# ============================================================
if mode == "Batch CSV":
    st.header("📂 Batch Clinical Audit")
    st.caption(
        "⚠️ **Known limitation:** batch-level scores can shift slightly "
        "depending on the size and composition of the uploaded file, due to "
        "how normalization layers behave across a group of patients scored "
        "together. This tool is intended for **research and retrospective "
        "audit** use. For individual patient risk assessment, use **Manual "
        "Entry**, which is unaffected by this and reflects the model's "
        "intended single-patient design."
    )
    uploaded = st.file_uploader("Upload Patient Records (CSV Format)", type=["csv"])

    if uploaded:
        df_raw = pd.read_csv(uploaded)
        df_raw.columns = df_raw.columns.str.strip('"').str.strip()
        st.success(f"📥 Loaded {len(df_raw)} patient records.")

        try:
            df_input = df_raw.copy()
            if 'ASAclassification' not in df_input.columns:
                st.error("🚨 'ASAclassification' column missing."); st.stop()

            # ── SMART IMPUTATION — surgery-specific means ──────────────
            # Load surgery-specific means from threshold_raw.json
            _surg_means    = thr_data.get("surgery_specific_means", {})
            _overall_means = thr_data.get("overall_training_means", {})
            _binary_cols   = thr_data.get("binary_feature_cols", [])

            _SURGERY_COLS = [
                "LapCholi","OpenCholi","Hernioplasty","Herniotomy","Lithotomy",
                "Pyeloplasty","Appendicectomy","Omentoplasty","SmallBowelResection",
                "Laproscopic LysisOfAdhesions","MRM","Hysterectomy","Prostectomy",
                "DiagLaprot","Nephrectomy","Gastrectomy","Oesophagotomy"
            ]

            # ASA encoding
            df_input['ASAclassification'] = (
                df_input['ASAclassification'].astype(str).str.strip())
            df_input['ASAclassification'] = apply_asa_encoding(
                df_input['ASAclassification'])

            # Reindex to feature_names with NaN for missing
            df_aligned = df_input.reindex(columns=feature_names, fill_value=np.nan)

            # Surgery-specific imputation row by row
            for _i in range(len(df_aligned)):
                # Detect surgery type for this patient
                _patient_surg = None
                for _s in _SURGERY_COLS:
                    if _s in df_aligned.columns:
                        _val = df_aligned.iloc[_i][_s]
                        if pd.notna(_val) and float(_val) == 1.0:
                            _patient_surg = _s
                            break

                # Choose imputation source
                _impute = (_surg_means.get(_patient_surg, _overall_means)
                           if _patient_surg else _overall_means)

                # Physiological range checks + fill missing
                _PHYSIO = {
                    "ALP":(5,2000),"PreOpTLC":(1000,80000),
                    "PostOpTLC":(500,80000),"PreOpSGOT":(5,5000),
                    "PostOpSGOT":(5,5000),"PreopCreat":(0.1,20),
                    "PostopCreat":(0.1,30),"PreopSodium":(100,170),
                    "PostOpSodium":(100,170),
                }
                for _col in feature_names:
                    _loc = df_aligned.columns.get_loc(_col)
                    _cur = df_aligned.iloc[_i, _loc]
                    if pd.notna(_cur) and _col in _PHYSIO:
                        _lo, _hi = _PHYSIO[_col]
                        if float(_cur) < _lo or float(_cur) > _hi:
                            df_aligned.iloc[_i, _loc] = np.nan
                    if pd.isna(df_aligned.iloc[_i, _loc]):
                        if _col == "ASAclassification":
                            df_aligned.iloc[_i, _loc] = 1.0
                        elif _col in _binary_cols:
                            df_aligned.iloc[_i, _loc] = 0.0
                        else:
                            df_aligned.iloc[_i, _loc] =                                 float(_impute.get(_col, 0.0))

            X_np = df_aligned.values.astype(np.float32)

            # ── CORRECT SCALER APPLICATION using feature_names_in_ ────
            if scaler is not None and len(SCALABLE_INDICES) > 0:
                _scaler_feats = (list(scaler.feature_names_in_)
                                 if hasattr(scaler, "feature_names_in_")
                                 else scalable_columns)
                _X_sc_df  = pd.DataFrame(X_np[:, SCALABLE_INDICES],
                                         columns=scalable_columns)
                _X_sc_df  = _X_sc_df[_scaler_feats]
                _X_scaled = scaler.transform(_X_sc_df)
                for _i, _feat in enumerate(scalable_columns):
                    _sp = _scaler_feats.index(_feat)
                    X_np[:, SCALABLE_INDICES[_i]] = _X_scaled[:, _sp]

            st.info("✅ Data aligned with clinical feature set.")
        except Exception as e:
            st.error(f"🚨 Preprocessing Failure: {e}"); st.stop()

        with st.spinner(f"🚀 Running {MC_RUNS} Monte Carlo passes × 4 models..."):
            m_means, gated_scores, uncertainties, entropy, entropy_norm, triage_levels = \
                load_models_and_mc_for_batch(X_np, n_forward_passes=MC_RUNS)

        results = df_raw.copy()
        results["P_VAE"]           = np.round(m_means[:, 0], 4)
        results["P_M1_Flipout"]    = np.round(m_means[:, 1], 4)
        results["P_M2_corrected"]  = np.round(m_means[:, 2], 4)
        results["P_Bayesian"]      = np.round(m_means[:, 3], 4)
        results["Ensemble_Mean"]   = np.round(np.mean(m_means, axis=1), 4)
        results["Uncertainty_SD"]  = np.round(uncertainties, 4)
        results["Entropy_Norm"]    = np.round(entropy_norm, 4)
        results["Gated_Score"]     = np.round(gated_scores, 4)
        runtime_thr = best_threshold_saved   # 0.4001 — Youden threshold
        results["Risk_Label"]      = np.where(gated_scores >= runtime_thr, "High Risk", "Low Risk")
        triage_display = [
            t.replace("🔴 ","").replace("🟡 ","").replace("🟢 ","")
            for t in triage_levels]
        results["Triage_Level"]    = triage_display
        results["Threshold_Used"]  = round(float(best_threshold_saved), 4)
        results["Threshold_Method"] = THRESHOLD_METHOD

        st.divider()
        st.write(f"📊 **Batch Diagnostics:** Avg Entropy: {float(np.mean(entropy)):.4f} | "
                 f"T_screen (Youden): {runtime_thr:.4f} | HIGH_RISK: {HIGH_RISK_BOUNDARY:.4f}")

        high_risk_count = int(np.sum(results["Risk_Label"] == "High Risk"))
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Sample",     len(results))
        m2.metric("High Risk Alerts", high_risk_count,
                  delta=f"{(high_risk_count/len(results)):.1%}", delta_color="inverse")
        m3.metric("Ensemble AUC", str(thr_data.get("ensemble_auc","—")))

        with st.expander("🔬 Calibration at Decision Threshold"):
            thr_runtime = thr_data.get("best_t_raw", best_threshold_saved)
            p_platt_thr = thr_data.get("calib_platt_at_thr")
            p_iso_thr   = thr_data.get("calib_iso_at_thr")
            p_beta_thr  = thr_data.get("calib_beta_at_thr")
            cs1, cs2, cs3 = st.columns(3)
            cs1.metric("Platt (PRIMARY)",
                       f"{p_platt_thr:.1%}" if p_platt_thr is not None else "n/a",
                       delta="✓ recommended", delta_color="off")
            cs2.metric("Isotonic",
                       f"{p_iso_thr:.1%}" if p_iso_thr is not None else "n/a",
                       delta="step-function artifact", delta_color="off")
            cs3.metric("Beta",
                       f"{p_beta_thr:.1%}" if p_beta_thr is not None else "n/a",
                       delta="parametric", delta_color="off")

        death_col = next(
            (c for c in results.columns
             if c.lower().strip('"').strip() in ("death","true_outcome","outcome","mortality")),
            None)
        if show_confusion and death_col:
            with st.expander("📊 Performance Metrics & ROC Curve"):
                raw_gt    = results[death_col].astype(str).str.strip('"').str.strip()
                gt_series = pd.to_numeric(raw_gt, errors="coerce")
                valid_mask = gt_series.notna()
                if valid_mask.sum() == 0:
                    st.warning(f"No valid numeric values in '{death_col}'.")
                else:
                    y_true_batch = gt_series[valid_mask].astype(int).values
                    y_pred_batch = (gated_scores[valid_mask] >= runtime_thr).astype(int)
                    cm   = confusion_matrix(y_true_batch, y_pred_batch)
                    rec  = recall_score(y_true_batch, y_pred_batch, zero_division=0)
                    prec = precision_score(y_true_batch, y_pred_batch, zero_division=0)
                    f1   = f1_score(y_true_batch, y_pred_batch, zero_division=0)
                    spec = cm[0,0]/(cm[0,0]+cm[0,1]) if (cm[0,0]+cm[0,1])>0 else 0.0
                    tn, fp, fn, tp = cm.ravel() if cm.size==4 else (cm[0,0],0,0,cm[1,1])
                    mc1,mc2,mc3,mc4 = st.columns(4)
                    mc1.metric("Sensitivity", f"{rec:.1%}",
                               delta="FN=0 ✅" if fn==0 else f"FN={fn} ⚠️",
                               delta_color="off" if fn==0 else "inverse")
                    mc2.metric("Specificity", f"{spec:.1%}", delta=f"FP={fp}")
                    mc3.metric("Precision",   f"{prec:.1%}")
                    mc4.metric("F1 Score",    f"{f1:.3f}")
                    st.markdown("**Confusion Matrix**")
                    st.dataframe(pd.DataFrame(cm,
                        index=["True: Survivor","True: Death"],
                        columns=["Pred: Safe","Pred: Flagged"]),
                        use_container_width=True)
                    n_pos = int(y_true_batch.sum()); n_neg = int(len(y_true_batch)-n_pos)
                    if n_pos >= 2 and n_neg >= 1:
                        auc = roc_auc_score(y_true_batch, gated_scores[valid_mask])
                        fpr, tpr, roc_thresholds = roc_curve(y_true_batch, gated_scores[valid_mask])
                        j_scores = tpr-fpr; best_idx = int(np.argmax(j_scores))
                        fig, ax = plt.subplots(figsize=(5,4))
                        ax.plot(fpr,tpr,color="#1a73e8",lw=2,label=f"Ensemble (AUC={auc:.3f})")
                        ax.plot([0,1],[0,1],"k--",lw=1,label="Random")
                        ax.scatter([fpr[best_idx]],[tpr[best_idx]],color="#d93025",zorder=5,
                                   label=f"Youden (thr≈{roc_thresholds[best_idx]:.3f})")
                        ax.set_xlabel("1 − Specificity (FPR)"); ax.set_ylabel("Sensitivity (TPR)")
                        ax.set_title("ROC Curve — 4-Model Ensemble")
                        ax.legend(fontsize=9); ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
                        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
                        a1,a2,a3 = st.columns(3)
                        a1.metric("AUC-ROC",f"{auc:.4f}",
                                  delta="excellent" if auc>=0.90 else "good" if auc>=0.80 else "fair")
                        a2.metric("Deaths",   f"{n_pos}")
                        a3.metric("Survivors",f"{n_neg}")
                    if n_pos>=1 and n_neg>=1:
                        plot_reliability(y_true_batch, gated_scores[valid_mask],
                                         "Reliability Curve — 4-Model Ensemble")

        st.subheader("📋 Detailed Clinical Triage List")
        def style_triage_row(val):
            text = str(val).upper()
            if "CRITICAL"  in text: return 'color:white;background-color:#d93025;font-weight:bold;'
            if "GRAY ZONE" in text: return 'color:black;background-color:#f9ab00;font-weight:bold;'
            if "SAFE"      in text: return 'color:white;background-color:#1e8e3e;font-weight:bold;'
            return ''
        fmt = {"Ensemble_Mean":"{:.4f}","Uncertainty_SD":"{:.4f}",
               "Entropy_Norm":"{:.4f}","Gated_Score":"{:.4f}"}
        styled_df = (results.style
                     .applymap(style_triage_row, subset=["Triage_Level"])
                     .background_gradient(subset=["Gated_Score"], cmap="YlOrRd")
                     .format(fmt))
        st.dataframe(styled_df, use_container_width=True)
        st.markdown("---")
        col_dl, col_gs = st.columns(2)
        with col_dl:
            csv_report = results.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="📥 Download Triage Report (CSV)",
                data=csv_report,
                file_name=f"clinical_triage_4model_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv")
        with col_gs:
            if st.button("🚀 Archive Batch to Clinical Vault"):
                with st.spinner("Syncing..."):
                    success, message = append_to_gsheet(results)
                    if success: st.success(f"Archived {len(results)} records.")
                    else: st.error(f"Sync failed: {message}")

# ============================================================
# 21b. MANUAL ENTRY
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
                        ["ASA-E","ASA_one","ASA_two","ASA_three","ASA_four"])
                elif f in categorical_features:
                    manual_data[f] = 1.0 if st.checkbox(f, value=False) else 0.0
                else:
                    manual_data[f] = st.number_input(f, value=0.0)
        submitted = st.form_submit_button("Predict")

    if submitted:
        df_manual = pd.DataFrame([manual_data])
        df_manual["ASAclassification"] = apply_asa_encoding(df_manual["ASAclassification"])
        X_manual = apply_training_scaling(df_manual)

        m_means, gated_scores, uncertainties, entropy, entropy_norm, triage_levels = \
            load_models_and_mc_for_batch(X_manual, n_forward_passes=MC_RUNS,
                                          use_frozen_batchnorm=True)

        current_triage    = triage_levels[0]
        adj_p             = float(gated_scores[0])
        e_val             = float(entropy[0])
        en_val            = float(entropy_norm[0])
        ensemble_mean_val = float(np.mean(m_means[0]))

        runtime_thr  = best_threshold_saved   # 0.4001 Youden threshold
        is_high_risk = adj_p >= runtime_thr
        is_near_miss = (not is_high_risk) and (adj_p >= best_threshold_saved - 0.10)
        label        = "High Risk" if is_high_risk else ("Borderline" if is_near_miss else "Low Risk")

        cal_p, cal_method = calibrate_single(adj_p, "Platt")

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
            st.caption(f"Threshold [{THRESHOLD_METHOD}]: {best_threshold_saved:.4f} | "
                       f"Entropy: {e_val:.4f}"
                       + (f" | Norm: {en_val:.4f}" if en_val > 0
                          else " | Norm: N/A (single patient — batch norm not applicable)"))
        with res_col2:
            st.metric("Calibrated Risk (Platt)", f"{cal_p:.1%}")
            st.caption(f"Platt scaling — primary calibrator at n_events=13. "
                       + (f"Youden J={J_SCORE:.4f}" if J_SCORE else
                          f"Threshold: {best_threshold_saved:.4f}"))

        st.markdown("### 🔬 Calibrated Mortality Risk Estimates")
        p_platt, _ = calibrate_single(adj_p, "Platt")
        p_iso,   _ = calibrate_single(adj_p, "Isotonic")
        p_beta,  _ = calibrate_single(adj_p, "Beta")
        col1, col2, col3 = st.columns(3)
        col1.metric("Platt", f"{p_platt:.1%}", delta="✓ RECOMMENDED", delta_color="off")
        col2.metric("Isotonic", f"{p_iso:.1%}",
                    delta="⚠ step-function artifact" if p_iso < p_platt*0.5 else "empirical",
                    delta_color="inverse" if p_iso < p_platt*0.5 else "off")
        col3.metric("Beta", f"{p_beta:.1%}",
                    delta="⚠ may overestimate" if p_beta > p_platt*2 else "parametric",
                    delta_color="inverse" if p_beta > p_platt*2 else "off")
        spread = max(p_platt,p_iso,p_beta) - min(p_platt,p_iso,p_beta)
        if spread > 0.15:
            st.warning(f"⚠️ Wide calibration spread ({spread:.0%}) — reflects small training "
                       f"sample (n=13 deaths). Platt is the most reliable estimate.")
        else:
            st.success(f"✅ Calibrators agree within {spread:.0%}.")
        st.caption("Triage zone (SAFE / GRAY ZONE / CRITICAL) is based on gated score vs threshold, "
                   "not on calibrated probability.")

        if adj_p < best_threshold_saved * 0.85:
            triage_context = "Score is well below the clinical threshold — very low risk."
        elif adj_p < best_threshold_saved:
            triage_context = "Score is approaching the threshold. Monitor closely."
        else:
            triage_context = "Score exceeds the clinical decision threshold — prioritise review."
        st.markdown(f"> **Clinical Context:** {triage_context}")
        if "CRITICAL" in current_triage:
            st.error(f"🚨 Score `{adj_p:.4f}` in critical zone. Immediate intervention advised.")
        elif "GRAY ZONE" in current_triage:
            st.warning(f"⚠️ Score `{adj_p:.4f}` elevated. Maintain high vigilance.")
        elif is_near_miss:
            st.info("💡 Borderline: within 10% of threshold. Review secondary risk factors.")

        st.divider()
        st.markdown("### 🤝 Model Committee Consensus (4 models — Performance-Normalized Weights)")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("VAE (DNN)",           f"{m_means[0, 0]:.6f}", delta=f"w={VAE_WEIGHT:.4f}", delta_color="off")
        mc2.metric("Flipout M1",          f"{m_means[0, 1]:.6f}", delta=f"w={M1_WEIGHT:.4f}", delta_color="off")
        mc3.metric("Probabilistic M2",    f"{m_means[0, 2]:.6f}", delta=f"w={M2_WEIGHT:.4f}", delta_color="off")
        mc4.metric("Bayesian MC",         f"{m_means[0, 3]:.6f}", delta=f"w={BAY_WEIGHT:.4f}", delta_color="off")

        v_vae = int(m_means[0,0] > thr_data.get("vote_threshold_vae", 0.6259))
        v_m1  = int(m_means[0,1] > thr_data.get("vote_threshold_mid", 0.4830))
        v_m2  = int(m_means[0,2] > thr_data.get("vote_threshold_m2",  0.6086))
        v_bay = int(m_means[0,3] > thr_data.get("vote_threshold_bay", 0.6077))
        votes_cast = v_vae + v_m1 + v_m2 + v_bay
        st.caption(f"**Majority gate ({MAJORITY_VOTES}/4 required):** "
                   f"{votes_cast}/4 models voted | "
                   f"Gate {'OPEN ✅' if votes_cast >= MAJORITY_VOTES else 'CLOSED ⛔'}")

        if show_uncertainty:
            st.markdown("#### 📉 Uncertainty")
            u1, u2, u3 = st.columns(3)
            u1.metric("Ensemble Std Dev", f"{float(uncertainties[0]):.4f}")
            u2.metric("Shannon Entropy",  f"{e_val:.4f}")
            u3.metric("Entropy (norm)",
                      f"{en_val:.4f}" if en_val > 0 else "N/A",
                      delta="single-patient — batch norm not applicable" if en_val==0 else None,
                      delta_color="off")

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
            "timestamp"       : pd.Timestamp.now().isoformat(),
            # ── Individual model scores — previously computed and shown
            # on-screen (Model Committee Consensus tiles) but never
            # persisted to the vault. "ensemble_mean" is the genuine
            # overall average of all 4 models (matches its existing
            # column meaning); "m2_prob" has no pre-existing column in
            # the sheet, so append_to_gsheet will auto-create one. ──
            "vae_prob"        : float(m_means[0, 0]),
            "flipout_prob"    : float(m_means[0, 1]),
            "m2_prob"         : float(m_means[0, 2]),
            "bayesian_prob"   : float(m_means[0, 3]),
            "ensemble_mean"   : ensemble_mean_val,
        })
        ok, err = append_to_gsheet(pd.DataFrame([audit_row]))
        if ok: st.success("✅ Patient record synced to clinical vault.")
        else: st.error(f"Sync failed: {err}")


# ============================================================
# 21c. PREOPERATIVE ASSESSMENT (separate model, see module above)
# ============================================================
elif mode == "Preoperative Assessment":
    run_preop_assessment_ui()
