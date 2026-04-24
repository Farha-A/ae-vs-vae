"""
Configuration and hyperparameters for the Medical MNIST denoising experiment.
"""
import os

# ─── Image & Training Hyperparameters ────────────────────────────────────────
IMG_SIZE = 64
BATCH_SIZE = 32
LATENT_DIM = 32
EPOCHS = 2
NOISE_LEVEL = 0.2

# ─── Dataset ─────────────────────────────────────────────────────────────────
# Local Dataset Path
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "medical-mnist")

# ─── Derived ─────────────────────────────────────────────────────────────────
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 1)


def get_classes(dataset_path: str | None = None) -> list[str]:
    """Return sorted list of class subdirectory names found in *dataset_path*."""
    path = dataset_path or DATASET_PATH
    return sorted(os.listdir(path))
