"""
visualize – Plotting utilities for latent-space, losses, denoising,
            reconstruction comparison, and **sample generation**.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.decomposition import PCA

from config import IMG_SIZE, LATENT_DIM, DATASET_PATH
from utils import add_noise, create_labeled_dataset, create_evaluation_dataset


# ═════════════════════════════════════════════════════════════════════════════
#  1. LOSS CURVES
# ═════════════════════════════════════════════════════════════════════════════

def plot_ae_losses(histories: dict):
    """Plot MSE training loss for every Denoising-AE region."""
    num_regions = len(histories)
    fig, axs = plt.subplots(num_regions, 1, figsize=(10, 4 * num_regions))
    plt.subplots_adjust(hspace=0.4)

    for i, (region, metrics) in enumerate(histories.items()):
        ax = axs[i] if num_regions > 1 else axs
        epochs = range(1, len(metrics["loss"]) + 1)

        ax.plot(epochs, metrics["loss"], "k-", label="MSE Loss", linewidth=2)

        ax.set_title(f"Denoising AE Loss: {region}")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.show()


def plot_vae_losses(histories: dict):
    """Plot total / reconstruction / KL loss per region."""
    num_regions = len(histories)
    fig, axs = plt.subplots(num_regions, 1, figsize=(10, 5 * num_regions))
    plt.subplots_adjust(hspace=0.4)

    for i, (region, metrics) in enumerate(histories.items()):
        ax = axs[i] if num_regions > 1 else axs
        epochs = range(1, len(metrics["loss"]) + 1)

        ax.plot(epochs, metrics["loss"], "k-", label="Total Loss", linewidth=2)
        ax.plot(epochs, metrics["reconstruction_loss"], "r--", label="Reconstruction Loss")
        ax.plot(epochs, metrics["kl_loss"], "b:", label="KL Divergence")

        ax.set_title(f"Denoising VAE Loss Breakdown: {region}")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss Value")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

    plt.show()


# ═════════════════════════════════════════════════════════════════════════════
#  2. LATENT-SPACE VISUALISATION (PCA → 2-D)
# ═════════════════════════════════════════════════════════════════════════════

def plot_latent_space(
    region_name: str,
    model_type: str,              # "ae" or "vae"
    ae_models: dict,
    vae_models: dict,
    classes: list[str],
    dataset_path: str = DATASET_PATH,
):
    """
    Project encodings of 500 random images (all classes) into 2-D via PCA
    and scatter-plot them, coloured by class.
    """
    labeled_ds = create_labeled_dataset(dataset_path, classes, batch_size=500)
    images, labels = next(iter(labeled_ds))
    images = images.numpy()
    labels = labels.numpy()

    if model_type == "ae":
        dae_instance = ae_models[region_name]
        latent_vectors = dae_instance.encoder.predict(images, verbose=0)
    else:
        v_enc, _ = vae_models[region_name]
        latent_vectors, _, _ = v_enc.predict(images, verbose=0)

    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_vectors)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap="viridis", alpha=0.6
    )
    cbar = plt.colorbar(scatter, ticks=range(len(classes)))
    cbar.ax.set_yticklabels(classes)
    plt.title(f"2D Latent Space ({model_type.upper()}) trained on {region_name}")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.show()


# ═════════════════════════════════════════════════════════════════════════════
#  3. DENOISING COMPARISON
# ═════════════════════════════════════════════════════════════════════════════

def plot_denoising_results(
    region_name: str,
    ae_models: dict,
    vae_models: dict,
    dataset_path: str = DATASET_PATH,
    noise_level: float = 0.3,
):
    """Original → Noisy → AE denoised → VAE denoised (side-by-side)."""
    eval_ds = create_evaluation_dataset(dataset_path, region_name, batch_size=1)
    clean_img = next(iter(eval_ds))           # shape (1, H, W, 1)

    noisy_img = add_noise(clean_img, noise_factor=noise_level)

    # AE
    ae_denoised = ae_models[region_name](noisy_img, training=False)

    # VAE
    v_enc, v_dec = vae_models[region_name]
    _, _, z = v_enc(noisy_img, training=False)
    vae_denoised = v_dec(z, training=False)

    imgs = [clean_img[0], noisy_img[0], ae_denoised[0], vae_denoised[0]]
    titles = ["Original", f"Noisy (σ={noise_level})", "AE Denoised", "VAE Denoised"]

    plt.figure(figsize=(15, 5))
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.imshow(imgs[i], cmap="gray")
        plt.title(titles[i])
        plt.axis("off")
    plt.suptitle(f"Denoising — {region_name}", fontsize=14)
    plt.tight_layout()
    plt.show()


# ═════════════════════════════════════════════════════════════════════════════
#  4. RECONSTRUCTION COMPARISON GRID
# ═════════════════════════════════════════════════════════════════════════════

def plot_reconstruction_comparison(
    classes: list[str],
    ae_models: dict,
    vae_models: dict,
    dataset_path: str = DATASET_PATH,
):
    """2 samples × 5 columns (Original, AE Latent, AE Recon, VAE Latent, VAE Recon) for every class."""
    num_classes = len(classes)
    fig, axs = plt.subplots(num_classes * 2, 5, figsize=(20, 3 * num_classes * 2))
    plt.subplots_adjust(hspace=0.4)

    for i, region in enumerate(classes):
        eval_ds = create_evaluation_dataset(dataset_path, region, batch_size=2)
        batch_imgs = next(iter(eval_ds))    # (2, H, W, 1)

        ae_model = ae_models[region]
        v_enc, v_dec = vae_models[region]

        for j in range(2):
            img = batch_imgs[j : j + 1]
            row_idx = i * 2 + j

            # Extract latent representations
            ae_latent = ae_model.encoder.predict(img, verbose=0)
            ae_rec = ae_model.decoder.predict(ae_latent, verbose=0)

            vae_latent, _, _ = v_enc.predict(img, verbose=0)
            vae_rec = v_dec.predict(vae_latent, verbose=0)

            # Determine shape for latent visualization
            latent_dim = ae_latent.shape[1]
            cols = int(np.ceil(np.sqrt(latent_dim)))
            rows = int(np.ceil(latent_dim / cols))
            
            # Original
            axs[row_idx, 0].imshow(img[0, :, :, 0], cmap="gray")
            axs[row_idx, 0].set_title(f"{region} (Orig) #{j + 1}")
            axs[row_idx, 0].axis("off")

            # AE Latent
            padded_ae_latent = np.pad(ae_latent[0], (0, rows * cols - latent_dim))
            axs[row_idx, 1].imshow(padded_ae_latent.reshape(rows, cols), cmap="viridis", interpolation='nearest')
            axs[row_idx, 1].set_title(f"{region} (AE Latent)")
            axs[row_idx, 1].axis("off")

            # AE Reconstruction
            axs[row_idx, 2].imshow(ae_rec[0, :, :, 0], cmap="gray")
            axs[row_idx, 2].set_title(f"{region} (AE Recon)")
            axs[row_idx, 2].axis("off")

            # VAE Latent
            padded_vae_latent = np.pad(vae_latent[0], (0, rows * cols - latent_dim))
            axs[row_idx, 3].imshow(padded_vae_latent.reshape(rows, cols), cmap="viridis", interpolation='nearest')
            axs[row_idx, 3].set_title(f"{region} (VAE Latent)")
            axs[row_idx, 3].axis("off")

            # VAE Reconstruction
            axs[row_idx, 4].imshow(vae_rec[0, :, :, 0], cmap="gray")
            axs[row_idx, 4].set_title(f"{region} (VAE Recon)")
            axs[row_idx, 4].axis("off")


# ═════════════════════════════════════════════════════════════════════════════

def plot_generated_samples(
    decoder,
    region_name: str,
    latent_dim: int = LATENT_DIM,
    n_samples: int = 16,
):
    """
    Generate novel images by sampling z ~ N(0, I) and decoding.

    This is only meaningful for VAE (not a vanilla AE) because the VAE's
    latent space is regularised to match a standard normal prior.
    """
    z_samples = tf.random.normal(shape=(n_samples, latent_dim))
    generated = decoder.predict(z_samples, verbose=0)

    cols = int(np.ceil(np.sqrt(n_samples)))
    rows = int(np.ceil(n_samples / cols))

    fig, axs = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    for idx in range(rows * cols):
        r, c = divmod(idx, cols)
        ax = axs[r, c] if rows > 1 else axs[c]
        if idx < n_samples:
            ax.imshow(generated[idx, :, :, 0], cmap="gray")
        ax.axis("off")

    plt.suptitle(f"Generated Samples (VAE) — {region_name}", fontsize=14)
    plt.tight_layout()
    plt.show()
