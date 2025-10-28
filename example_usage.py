"""
Example usage of the Stable Diffusion components
"""

import torch
from sd import (
    VAE_Encoder,
    VAE_Decoder,
    CLIP,
    DiffusionUNET,
    DiffusionScheduler
)


def demonstrate_vae():
    """Demonstrate VAE encoding and decoding"""
    print("=== VAE Encoder/Decoder Demo ===")
    
    # Create models
    encoder = VAE_Encoder()
    decoder = VAE_Decoder()
    
    # Create dummy image (Batch_Size=1, Channels=3, Height=512, Width=512)
    dummy_image = torch.randn(1, 3, 512, 512)
    dummy_noise = torch.randn(1, 4, 64, 64)  # Latent space noise
    
    print(f"Original image shape: {dummy_image.shape}")
    
    # Encode to latent space
    latent = encoder(dummy_image, dummy_noise)
    print(f"Latent shape: {latent.shape}")
    
    # Decode back to image
    reconstructed = decoder(latent)
    print(f"Reconstructed image shape: {reconstructed.shape}")
    print()


def demonstrate_clip():
    """Demonstrate CLIP text encoding"""
    print("=== CLIP Text Encoder Demo ===")
    
    # Create CLIP model
    clip_model = CLIP()
    
    # Create dummy tokens (Batch_Size=1, Seq_Len=77)
    dummy_tokens = torch.randint(0, 49408, (1, 77))
    
    print(f"Token shape: {dummy_tokens.shape}")
    
    # Encode text
    text_embedding = clip_model(dummy_tokens)
    print(f"Text embedding shape: {text_embedding.shape}")
    print()


def demonstrate_unet():
    """Demonstrate UNet diffusion model"""
    print("=== UNet Diffusion Model Demo ===")
    
    # Create UNet model
    unet = DiffusionUNET()
    
    # Create dummy inputs
    dummy_image = torch.randn(1, 4, 64, 64)  # Latent space
    dummy_context = torch.randn(1, 77, 768)  # CLIP embedding
    dummy_time = torch.tensor([500])  # Timestep
    
    print(f"Noisy image (latent) shape: {dummy_image.shape}")
    print(f"Context (CLIP) shape: {dummy_context.shape}")
    print(f"Time shape: {dummy_time.shape}")
    
    # Predict noise
    predicted_noise = unet(dummy_image, dummy_context, dummy_time)
    print(f"Predicted noise shape: {predicted_noise.shape}")
    print()


def demonstrate_diffusion_process():
    """Demonstrate the complete diffusion process"""
    print("=== Complete Diffusion Process Demo ===")
    
    # Create models
    unet = DiffusionUNET()
    scheduler = DiffusionScheduler(n_steps=100)
    
    # Create dummy context
    dummy_context = torch.randn(1, 77, 768)
    
    print(f"Context shape: {dummy_context.shape}")
    print(f"Number of diffusion steps: {scheduler.n_steps}")
    
    # Generate sample (this would take time with full 1000 steps)
    print("Starting diffusion sampling process...")
    with torch.no_grad():
        # Smaller shape for demo (64x64 latent)
        samples = scheduler.sample(unet, dummy_context, shape=(1, 4, 64, 64), device='cpu')
    print(f"Generated sample shape: {samples.shape}")
    print()


if __name__ == "__main__":
    print("Stable Diffusion PyTorch - Component Demos")
    print("=" * 50)
    print()
    
    # Run demos
    try:
        demonstrate_vae()
    except Exception as e:
        print(f"VAE demo failed: {e}")
    
    try:
        demonstrate_clip()
    except Exception as e:
        print(f"CLIP demo failed: {e}")
    
    try:
        demonstrate_unet()
    except Exception as e:
        print(f"UNet demo failed: {e}")
    
    try:
        demonstrate_diffusion_process()
    except Exception as e:
        print(f"Diffusion demo failed: {e}")
    
    print("=" * 50)
    print("All demos completed!")

