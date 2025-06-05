# streamlit_app.py

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import joblib
import json
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler

tfd = tfp.distributions
tfpl = tfp.layers

# Load artifacts
scaler, columns_to_scale = joblib.load("models/scaler.pkl")
feature_names = joblib.load("models/feature_names.pkl")
iso_reg = joblib.load("models/iso_reg.pkl")

with open('best_threshold.json', 'r') as f:
    best_threshold = json.load(f)['best_threshold']

# Register custom objects
@tf.keras.utils.register_keras_serializable()
def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfd.MultivariateNormalDiag(loc=t, scale_diag=tf.ones_like(t)))
    ])

@tf.keras.utils.register_keras_serializable()
def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(tfp.layers.IndependentNormal.params_size(n), dtype=dtype),
        tfp.layers.IndependentNormal(n, convert_to_tensor_fn=tfd.Distribution.sample),
    ])

def negative_log_likelihood(true_labels, predicted_labels):
    true_labels = tf.cast(true_labels, tf.float32)
    dist = tfd.Normal(loc=predicted_labels, scale=1)
    return -tf.reduce_mean(dist.log_prob(true_labels))

custom_objects = {
    'DenseFlipout': tfpl.DenseFlipout,
    'DenseFlipoutLayer': tfpl.DenseFlipout,
    'DenseVariational': tfp.layers.DenseVariational,
    'prior': prior,
    'posterior': posterior,
    'KLDivergenceRegularizer': tfpl.KLDivergenceRegularizer,
    'negative_log_likelihood': negative_log_likelihood
}

# Load models
vae_model = tf.keras.models.load_model("models/vae_model.h5")
model_2 = tf.keras.models.load_model("models/model_2_probabilistic", custom_objects=custom_objects)
bayesian_model = tf.keras.models.load_model("models/bayesian_model", custom_objects=custom_objects)
ensemble_models = [vae_model, model_2, bayesian_model]

# Prediction function
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

# Streamlit UI
st.title("Ensemble Model Predictor")

uploaded_file = st.file_uploader("Upload Patient CSV File", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)

    st.write("Uploaded Data:")
    st.dataframe(input_df)

    # Validate input
    missing_cols = set(feature_names) - set(input_df.columns)
    if missing_cols:
        st.error(f"Missing required fields: {missing_cols}")
    else:
        input_df = input_df[feature_names].fillna(0)
        input_df[columns_to_scale] = scaler.transform(input_df[columns_to_scale])
        input_array = input_df.values

        all_probs = ensemble_models_predict_all(input_array, n_forward_passes=100)
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
