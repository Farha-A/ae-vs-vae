# Medical MNIST – Denoising Autoencoders & VAEs

Autoencoder (AE) and Variational Autoencoder (VAE) models trained on the
[Medical MNIST](https://www.kaggle.com/datasets/andrewmvd/medical-mnist) dataset.
The project explores data reconstruction, denoising, latent-space visualization,
and sample generation.

## Features

| Feature | Description |
| --- | --- |
| AE | Convolutional encoder-decoder for reconstruction |
| Denoising AE | Custom training step that injects Gaussian noise |
| VAE | Probabilistic latent space with reparameterization trick |
| Denoising VAE | VAE variant trained on noisy inputs |
| Latent-space visualization | 2-D PCA projection coloured by class |
| Loss visualization | MSE (AE) and reconstruction + KL divergence (VAE) |
| Denoising comparison | Side-by-side original → noisy → AE → VAE |
| Sample generation | Novel images sampled from the VAE prior |

## Project Structure

```bash
nmnt1/
├── config.py              # Hyperparameters & dataset path
├── main.py                # Full experiment entry-point
├── train.py               # Training loops (Denoising AE + Denoising VAE)
├── utils.py               # tf.data pipelines & helpers
├── visualize.py           # All plotting / visualization functions
├── models/
│   ├── __init__.py
│   ├── autoencoder.py     # AE & Denoising AE
│   └── vae.py             # Sampling layer, VAE & Denoising VAE
├── notebook.ipynb         # Interactive experiment notebook
├── medical-mnist/         # Dataset (git-ignored)
└── .gitignore
```

## Quick Start

1. **Install dependencies**

   ```bash
   pip install tensorflow matplotlib scikit-learn
   ```

2. **Place the dataset**
   Download [Medical MNIST](https://www.kaggle.com/datasets/andrewmvd/medical-mnist)
   and extract it into a `medical-mnist/` folder at the project root (the default
   path in `config.py`).

3. **Run the full pipeline**

   ```bash
   python main.py
   ```

   This will train both Denoising AE and Denoising VAE models for every region,
   then produce all visualizations (losses, reconstruction, denoising, latent
   space, sample generation).

4. **Or use the notebook**
   Open [`notebook.ipynb`](notebook.ipynb) for a step-by-step interactive
   walkthrough of the same experiments.

## Configuration

All hyperparameters live in [`config.py`](config.py):

| Parameter | Default | Description |
| --- | --- | --- |
| `IMG_SIZE` | 64 | Image resize target |
| `BATCH_SIZE` | 32 | Training batch size |
| `LATENT_DIM` | 32 | Dimensionality of the latent space |
| `EPOCHS` | 2 | Training epochs per region |
| `NOISE_LEVEL` | 0.2 | Gaussian noise σ injected during training |
| `DATASET_PATH` | `./medical-mnist` | Path to the extracted dataset |
