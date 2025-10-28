# Stable Diffusion PyTorch Implementation

A PyTorch implementation of the Stable Diffusion model for image generation.

## Overview

This project implements a complete Stable Diffusion model with the following components:

- **VAE (Variational Autoencoder)**: Encoder and Decoder for latent space operations
- **CLIP Text Encoder**: Text conditioning model
- **UNet Diffusion Model**: The main denoising network with cross-attention
- **Diffusion Scheduler**: DDPM-style noise scheduling

## Project Structure

```
diffusion_pytorch/
├── sd/
│   ├── __init__.py         # Package initialization
│   ├── attention.py         # Self-attention mechanism
│   ├── clip.py              # CLIP text encoder
│   ├── cross_attention.py   # Cross-attention for conditioning
│   ├── decoder.py           # VAE decoder
│   ├── encoder.py           # VAE encoder
│   ├── timestep_embedding.py # Sinusoidal positional embeddings
│   ├── unet.py              # UNet diffusion model
│   └── diffusion.py         # Diffusion scheduler
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Model Components

#### VAE Encoder
```python
from sd import VAE_Encoder

encoder = VAE_Encoder()
# Encodes images to latent space
latent = encoder(image, noise)
```

#### VAE Decoder
```python
from sd import VAE_Decoder

decoder = VAE_Decoder()
# Decodes latents back to images
image = decoder(latent)
```

#### CLIP Text Encoder
```python
from sd import CLIP

clip_model = CLIP()
# Encodes text prompts
text_embedding = clip_model(tokens)
```

#### UNet Diffusion Model
```python
from sd import DiffusionUNET

unet = DiffusionUNET()
# Predicts noise given noisy image, context, and time
predicted_noise = unet(noisy_image, context, timestep)
```

#### Complete Diffusion Pipeline
```python
from sd import DiffusionUNET, DiffusionScheduler

# Initialize models
unet = DiffusionUNET()
scheduler = DiffusionScheduler()

# Sample with DDPM
samples = scheduler.sample(unet, context, shape=(1, 4, 64, 64))
```

## Model Architecture

### VAE
- Encoder: Down-samples RGB images (3 channels) to latent space (4 channels)
- Decoder: Up-samples latent space (4 channels) back to RGB images (3 channels)

### CLIP Encoder
- 12 transformer layers with self-attention
- Embedding dimension: 768
- Vocabulary size: 49,408
- Context length: 77 tokens

### UNet
- U-shaped architecture with encoder-decoder structure
- Cross-attention blocks for text conditioning
- Self-attention for spatial relationships
- Time embeddings for diffusion timesteps

## Training

To train the model, you would typically:

1. Prepare your dataset
2. Use the VAE encoder to compress images to latent space
3. Use CLIP to encode text prompts
4. Train the UNet to predict noise using the diffusion loss
5. Use the decoder to convert generated latents back to images

## License

This implementation is for educational purposes.

