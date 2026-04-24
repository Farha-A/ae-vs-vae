"""
utils – Data-loading helpers built on tf.data and shared utility functions.

NOTE: The assignment requires tf.data pipelines rather than
      keras.preprocessing.image.ImageDataGenerator.
"""

import os

import tensorflow as tf

from config import IMG_SIZE, BATCH_SIZE


# ─── Image loading (used inside tf.data.map) ────────────────────────────────

def load_image(file_path):
    """Read, decode, resize and normalise a single image to [0, 1]."""
    raw = tf.io.read_file(file_path)
    img = tf.io.decode_image(raw, channels=1, expand_animations=False)
    img.set_shape([None, None, 1])           # static shape needed after decode_image
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    return img


# ─── Noise ───────────────────────────────────────────────────────────────────

def add_noise(images, noise_factor=0.2):
    """Add Gaussian noise to *images* and clip to [0, 1]."""
    noise = tf.random.normal(
        shape=tf.shape(images), mean=0.0, stddev=noise_factor, dtype=tf.float32
    )
    return tf.clip_by_value(images + noise, 0.0, 1.0)


# ─── Training dataset (single region, AE/VAE mode: input == target) ─────────

def create_train_dataset(
    dataset_path: str,
    region: str,
    batch_size: int = BATCH_SIZE,
) -> tf.data.Dataset:
    """
    Build a ``tf.data.Dataset`` of (image, image) pairs for *region*.

    The target is identical to the input so that the model learns
    reconstruction.  Shuffling and prefetching are enabled automatically.
    """
    pattern = os.path.join(dataset_path, region, "*")
    ds = tf.data.Dataset.list_files(pattern, shuffle=True)
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda x: (x, x))       # input == target for autoencoders
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ─── Labelled dataset (all regions, for latent-space visualisation) ──────────

def create_labeled_dataset(
    dataset_path: str,
    classes: list[str],
    batch_size: int = 500,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """
    Build a ``tf.data.Dataset`` of (image, int_label) pairs across every
    class listed in *classes*.  Useful for colour-coded latent-space plots.
    """
    all_files: list[str] = []
    all_labels: list[int] = []
    for i, cls in enumerate(classes):
        pattern = os.path.join(dataset_path, cls, "*")
        files = sorted(tf.io.gfile.glob(pattern))
        all_files.extend(files)
        all_labels.extend([i] * len(files))

    ds = tf.data.Dataset.from_tensor_slices((all_files, all_labels))
    if shuffle:
        ds = ds.shuffle(len(all_files))
    ds = ds.map(
        lambda f, l: (load_image(f), l),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ─── Single-region image batch (for evaluation / denoising demos) ────────────

def create_evaluation_dataset(
    dataset_path: str,
    region: str,
    batch_size: int = 1,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """Return single-region images (no labels) as a ``tf.data.Dataset``."""
    pattern = os.path.join(dataset_path, region, "*")
    ds = tf.data.Dataset.list_files(pattern, shuffle=shuffle)
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
