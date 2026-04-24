"""
train – Training loops for Denoising AE and Denoising VAE.

Run directly:
    python train.py
"""

import os

# Suppress noisy TF logs before importing tensorflow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow import keras

tf.get_logger().setLevel("ERROR")

from config import (
    IMG_SHAPE, LATENT_DIM, EPOCHS, NOISE_LEVEL, DATASET_PATH, get_classes,
)
from models.autoencoder import build_autoencoder_components, DenoisingAE
from models.vae import build_vae_components, DenoisingVAE
from utils import create_train_dataset


def train_denoising_autoencoders(
    classes: list[str],
    dataset_path: str = DATASET_PATH,
    epochs: int = EPOCHS,
) -> tuple[dict, dict]:
    """Train one DenoisingAE per region. Return (histories, models) dicts keyed by region."""
    ae_histories: dict = {}
    ae_models: dict = {}

    print("--- TRAINING DENOISING AUTOENCODERS ---")
    for region in classes:
        print(f"Region: {region}")
        train_ds = create_train_dataset(dataset_path, region)

        enc, dec = build_autoencoder_components(IMG_SHAPE, LATENT_DIM)
        dae = DenoisingAE(enc, dec, noise_factor=NOISE_LEVEL)
        dae.compile(optimizer="adam")

        history = dae.fit(train_ds, epochs=epochs, verbose=1)
        ae_histories[region] = history.history
        ae_models[region] = dae

    return ae_histories, ae_models


def train_denoising_vaes(
    classes: list[str],
    dataset_path: str = DATASET_PATH,
    epochs: int = EPOCHS,
) -> tuple[dict, dict]:
    """Train one DenoisingVAE per region. Return (histories, models) dicts keyed by region."""
    vae_histories: dict = {}
    vae_models: dict = {}

    print("--- STARTING DENOISING VAE TRAINING ---")
    for region in classes:
        print(f"\nTraining DVAE for: {region}")
        train_ds = create_train_dataset(dataset_path, region)

        enc, dec = build_vae_components(IMG_SHAPE, LATENT_DIM)
        dvae = DenoisingVAE(enc, dec, noise_factor=NOISE_LEVEL)
        dvae.compile(optimizer=keras.optimizers.Adam())

        history = dvae.fit(train_ds, epochs=epochs)
        vae_histories[region] = history.history
        vae_models[region] = (enc, dec)

    return vae_histories, vae_models


# ─── CLI entry-point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    classes = get_classes()
    print(f"Detected regions: {classes}")

    ae_histories, ae_models = train_denoising_autoencoders(classes)
    vae_histories, vae_models = train_denoising_vaes(classes)

    print("\n✓ Training complete.")
