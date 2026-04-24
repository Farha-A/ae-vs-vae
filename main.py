"""
main – Full experiment pipeline.

Trains Denoising AE and Denoising VAE on every Medical MNIST region,
then produces all required visualisations:
    1. Loss curves  (AE + VAE)
    2. Reconstruction comparison  (Original / AE / VAE)
    3. Denoising demo  (Original / Noisy / AE / VAE)
    4. Latent-space PCA plots  (AE + VAE, per region)
    5. Sample generation  (VAE, per region)

Run:
    python main.py
"""

from config import DATASET_PATH, get_classes
from train import train_denoising_autoencoders, train_denoising_vaes
from visualize import (
    plot_ae_losses,
    plot_vae_losses,
    plot_reconstruction_comparison,
    plot_denoising_results,
    plot_latent_space,
    plot_generated_samples,
)


def main():
    classes = get_classes()
    print(f"Detected regions: {classes}\n")

    # ── 1. Train models ──────────────────────────────────────────────────
    ae_histories, ae_models = train_denoising_autoencoders(classes)
    vae_histories, vae_models = train_denoising_vaes(classes)

    # ── 2. Loss visualisation ────────────────────────────────────────────
    print("\n── Loss Curves ──")
    plot_ae_losses(ae_histories)
    plot_vae_losses(vae_histories)

    # ── 3. Reconstruction comparison ─────────────────────────────────────
    print("\n── Reconstruction Comparison ──")
    plot_reconstruction_comparison(classes, ae_models, vae_models)

    # ── 4. Denoising demo ────────────────────────────────────────────────
    print("\n── Denoising Demo ──")
    for region in classes:
        plot_denoising_results(region, ae_models, vae_models)

    # ── 5. Latent-space visualisation ────────────────────────────────────
    print("\n── Latent Space ──")
    for region in classes:
        plot_latent_space(region, "ae", ae_models, vae_models, classes)
        plot_latent_space(region, "vae", ae_models, vae_models, classes)

    # ── 6. Sample generation (VAE only) ──────────────────────────────────
    print("\n── Sample Generation ──")
    for region in classes:
        _, dec = vae_models[region]
        plot_generated_samples(dec, region)

    print("\n✓ All experiments complete.")


if __name__ == "__main__":
    main()
