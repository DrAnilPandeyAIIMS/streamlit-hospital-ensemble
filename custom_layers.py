import tensorflow as tf
import tensorflow_probability as tfp

# Custom DenseFlipout Layer
class DenseFlipoutLayer(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super(DenseFlipoutLayer, self).__init__(**kwargs)
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        self.dense_flipout = tfp.layers.DenseFlipout(self.units, activation=self.activation)
        super(DenseFlipoutLayer, self).build(input_shape)

    def call(self, inputs):
        return self.dense_flipout(inputs)

# Correct Bernoulli Negative Log-Likelihood Loss
def negative_log_likelihood_bernoulli(y_true, logits):
    y_true = tf.cast(y_true, tf.float32)
    predicted_distribution = tfp.distributions.Bernoulli(logits=logits)
    return -tf.reduce_mean(predicted_distribution.log_prob(y_true))

# Build Bayesian Model using Flipout for all layers
def build_probabilistic_model(input_shape):
    model = tf.keras.Sequential([
        DenseFlipoutLayer(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        DenseFlipoutLayer(2048, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        DenseFlipoutLayer(1024, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        DenseFlipoutLayer(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        DenseFlipoutLayer(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        DenseFlipoutLayer(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),

        DenseFlipoutLayer(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),

        DenseFlipoutLayer(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),

        DenseFlipoutLayer(32, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),

        DenseFlipoutLayer(16, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),

        DenseFlipoutLayer(8, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),

        DenseFlipoutLayer(1)  # No activation (logits)
    ])
    return model
