# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import joblib
import json
from sklearn.isotonic import IsotonicRegression
import streamlit as st

# Configure the page
st.set_page_config(
    page_title="Hospital Ensemble Predictor",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("🏥 Hospital Mortality Predictor")
st.markdown("Upload patient data or enter values to get predictions.")

tfd = tfp.distributions
scaler = joblib.load("models/scaler.pkl")
feature_names = joblib.load("models/feature_names.pkl")
iso_reg = joblib.load("models/iso_reg.pkl")

# Load best threshold
with open("best_threshold.json", "r") as f:
    best_threshold = json.load(f)["best_threshold"]

# --- Custom prior ---
def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfd.MultivariateNormalDiag(
            loc=t,
            scale_diag=tf.ones_like(t)
        ))
    ])

# --- Custom posterior ---
def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(tfp.layers.IndependentNormal.params_size(n), dtype=dtype),
        tfp.layers.IndependentNormal(n, convert_to_tensor_fn=tfd.Distribution.sample),
    ])

# --- Custom loss ---
def negative_log_likelihood_bernoulli(y_true, logits):
    predicted_distribution = tfp.distributions.Bernoulli(logits=logits)
    return -tf.reduce_mean(predicted_distribution.log_prob(tf.cast(y_true, tf.float32)))

# --- Custom layer ---
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
        self.make_prior_fn = make_prior_fn
        self.make_posterior_fn = make_posterior_fn
        self.kl_weight = kl_weight

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "make_prior_fn": "prior",
            "make_posterior_fn": "posterior",
            "kl_weight": self.kl_weight
        })
        return config

    @classmethod
    def from_config(cls, config):
        config["make_prior_fn"] = prior
        config["make_posterior_fn"] = posterior
        return cls(**config)

custom_objects = {
    'CustomDenseVariational': CustomDenseVariational,
    'negative_log_likelihood': negative_log_likelihood_bernoulli, 
    'negative_log_likelihood_bernoulli': negative_log_likelihood_bernoulli,
    'prior': prior,
    'posterior': posterior
}

vae_model = tf.keras.models.load_model("models/vae_model.h5")
model_2 = tf.keras.models.load_model(
    "models/model_2_probabilistic",
    custom_objects={
        'DenseFlipoutLayer': tfp.layers.DenseFlipout,
        'DenseFlipout': tfp.layers.DenseFlipout,
        'negative_log_likelihood_bernoulli': negative_log_likelihood_bernoulli,
        'negative_log_likelihood': negative_log_likelihood_bernoulli
    }
)

bayesian_model = tf.keras.models.load_model("models/bayesian_model", custom_objects=custom_objects)
ensemble_models = [vae_model, model_2, bayesian_model]

def ensemble_models_predict_all(input_array, n_forward_passes=100):
    input_tensor = tf.convert_to_tensor(input_array, dtype=tf.float32)
    all_model_probs = []

    vae_probs = vae_model(input_tensor, training=False).numpy().flatten()
    vae_stack = np.stack([vae_probs] * n_forward_passes)
    all_model_probs.append(vae_stack)

    for model in [model_2, bayesian_model]:
        model_probs = []
        for _ in range(n_forward_passes):
            probs = model(input_tensor, training=True).numpy().flatten()
            model_probs.append(probs)
        all_model_probs.append(np.array(model_probs))

    all_model_probs = np.concatenate(all_model_probs, axis=0)
    return all_model_probs

st.title("Ensemble Model Predictor")

uploaded_file = st.file_uploader("Upload Patient CSV File", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)

    st.write("Uploaded Data:")
    st.dataframe(input_df)

    missing_cols = set(feature_names) - set(input_df.columns)
    if missing_cols:
        st.error(f"Missing required fields: {missing_cols}")
    else:
        input_df = input_df[feature_names].fillna(0)
        columns_to_scale = input_df.columns
        input_df[columns_to_scale] = scaler.transform(input_df[columns_to_scale])
        input_array = input_df.values

        all_probs = ensemble_models_predict_all(input_array)
        mean_probs = np.mean(all_probs, axis=0)
        std_devs = np.std(all_probs, axis=0)
        entropy = - (mean_probs * np.log2(mean_probs + 1e-9) + (1 - mean_probs) * np.log2(1 - mean_probs + 1e-9))

        calibrated_probs = iso_reg.predict(mean_probs.reshape(-1, 1)).flatten()
        predicted_labels = (calibrated_probs >= best_threshold).astype(int)

        results_df = input_df.copy()
        results_df['raw_probability'] = mean_probs
        results_df['calibrated_probability'] = calibrated_probs
        results_df['predicted_label'] = predicted_labels
        results_df['std_deviation'] = std_devs
        results_df['entropy'] = entropy

        st.subheader("Prediction Results")
        st.dataframe(results_df)

        st.download_button("Download Results as CSV", results_df.to_csv(index=False), "predictions.csv", "text/csv")

else:
    st.subheader("📋 Enter Patient Data Manually")
    with st.form("manual_form"):
        manual_data = {}
        for feature in feature_names:
            if feature.lower() in ["comorbidity", "on_ventilator", "diabetic", "hypertensive"]:
                manual_data[feature] = st.selectbox(f"{feature}", ["No", "Yes"])
            else:
                manual_data[feature] = st.number_input(f"{feature}", step=0.1)

        submitted = st.form_submit_button("Predict")

        if submitted:
            df_input = pd.DataFrame([manual_data])
            for col in df_input.columns:
                if df_input[col].dtype == object:
                    df_input[col] = df_input[col].map({"Yes": 1, "No": 0})
            df_input = df_input[feature_names].fillna(0)
            df_input[df_input.columns] = scaler.transform(df_input[df_input.columns])
            input_array = df_input.values

            all_probs = ensemble_models_predict_all(input_array)
            mean_probs = np.mean(all_probs, axis=0)
            std_devs = np.std(all_probs, axis=0)
            entropy = - (mean_probs * np.log2(mean_probs + 1e-9) + (1 - mean_probs) * np.log2(1 - mean_probs + 1e-9))

            calibrated_probs = iso_reg.predict(mean_probs.reshape(-1, 1)).flatten()
            predicted_labels = (calibrated_probs >= best_threshold).astype(int)

            st.subheader("Prediction Result")
            st.write(f"**Raw Probability:** {mean_probs[0]:.3f}")
            st.write(f"**Calibrated Probability:** {calibrated_probs[0]:.3f}")
            st.write(f"**Predicted Label:** {'High Risk' if predicted_labels[0] == 1 else 'Low Risk'}")
            st.write(f"**Uncertainty (Std Dev):** {std_devs[0]:.3f}")
            st.write(f"**Entropy:** {entropy[0]:.3f}")


