"""
Autoencoder architectures (regular and denoising).
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ─── Regular Autoencoder ─────────────────────────────────────────────────────

def build_autoencoder(img_shape, latent_dim):
    """Build and return a compiled convolutional autoencoder."""
    # Encoder
    encoder_inputs = keras.Input(shape=img_shape)
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    latent_space = layers.Dense(latent_dim)(x)
    encoder = keras.Model(encoder_inputs, latent_space)

    # Decoder
    decoder_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(16 * 16 * 64, activation="relu")(decoder_inputs)
    x = layers.Reshape((16, 16, 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
    decoder = keras.Model(decoder_inputs, decoder_outputs)

    autoencoder = keras.Model(encoder_inputs, decoder(encoder(encoder_inputs)))
    autoencoder.compile(optimizer="adam", loss="mse")

    return autoencoder


# ─── Denoising Autoencoder ───────────────────────────────────────────────────

def build_autoencoder_components(img_shape, latent_dim):
    """Return (encoder, decoder) as separate Keras models (used by DenoisingAE)."""
    # Encoder
    encoder_inputs = keras.Input(shape=img_shape)
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    latent_space = layers.Dense(latent_dim, name="latent_vector")(x)
    encoder = keras.Model(encoder_inputs, latent_space, name="ae_encoder")

    # Decoder
    decoder_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(16 * 16 * 64, activation="relu")(decoder_inputs)
    x = layers.Reshape((16, 16, 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
    decoder = keras.Model(decoder_inputs, decoder_outputs, name="ae_decoder")

    return encoder, decoder


class DenoisingAE(keras.Model):
    """Custom Keras model that injects Gaussian noise during training."""

    def __init__(self, encoder, decoder, noise_factor=0.2, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.noise_factor = noise_factor
        self.loss_tracker = keras.metrics.Mean(name="loss")

    def call(self, inputs, training=False):
        latent = self.encoder(inputs)
        return self.decoder(latent)

    def train_step(self, data):
        if isinstance(data, tuple):
            clean_images = data[0]

        noise = tf.random.normal(
            shape=tf.shape(clean_images), mean=0.0, stddev=self.noise_factor
        )
        noisy_images = tf.clip_by_value(clean_images + noise, 0.0, 1.0)

        with tf.GradientTape() as tape:
            reconstruction = self(noisy_images, training=True)
            loss = tf.reduce_mean(tf.square(clean_images - reconstruction))

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
