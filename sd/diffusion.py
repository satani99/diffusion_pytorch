import torch
from torch import nn
import math


class DiffusionScheduler:
    """DDPM-style scheduler for diffusion process"""
    
    def __init__(self, n_steps=1000, beta_start=0.00085, beta_end=0.0120):
        self.n_steps = n_steps
        
        # Create a linear schedule for beta values
        self.beta = torch.linspace(beta_start, beta_end, n_steps)
        self.alpha = 1.0 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)
        
    def add_noise(self, x0, t, noise):
        """Add noise to the original image at timestep t"""
        sqrt_alpha_cumprod_t = torch.sqrt(self.alpha_cumprod[t])[:, None, None, None]
        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1.0 - self.alpha_cumprod[t])[:, None, None, None]
        
        return sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise
    
    def get_velocity(self, x0, noise, t):
        """Compute velocity prediction target"""
        sqrt_alpha_cumprod_t = torch.sqrt(self.alpha_cumprod[t])[:, None, None, None]
        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1.0 - self.alpha_cumprod[t])[:, None, None, None]
        
        return sqrt_alpha_cumprod_t * noise - sqrt_one_minus_alpha_cumprod_t * x0
    
    @torch.no_grad()
    def sample(self, model, context, shape, device='cuda'):
        """Sample from the diffusion model using DDPM"""
        x = torch.randn(shape, device=device)
        
        for t in range(self.n_steps - 1, -1, -1):
            # Get current timestep
            t_tensor = torch.tensor([t], device=device).expand(shape[0])
            
            # Predict noise
            predicted_noise = model(x, context, t_tensor)
            
            # Compute alpha and beta values for current step
            alpha_t = self.alpha[t].to(device)
            alpha_cumprod_t = self.alpha_cumprod[t].to(device)
            beta_t = self.beta[t].to(device)
            
            # Predict x0 from current noisy image
            pred_x0 = (x - torch.sqrt(1.0 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
            
            # Direction pointing to x_t
            pred_dir = torch.sqrt(beta_t) * predicted_noise
            
            if t > 0:
                x = torch.sqrt(alpha_t) * pred_x0 + torch.sqrt(1.0 - alpha_t) * torch.randn_like(x)
            else:
                x = pred_x0
                
        return x


class Diffusion(nn.Module):
    """Main diffusion model"""
    
    def __init__(self):
        super().__init__()
        self.model = None  # Will be set externally
        self.scheduler = DiffusionScheduler()
    
    def set_model(self, model):
        """Set the UNet model"""
        self.model = model
    
    def forward(self, x, context, t):
        """Forward pass through the diffusion model"""
        return self.model(x, context, t)


def get_time_embedding(timesteps, dim=320):
    """Create sinusoidal time embeddings"""
    device = timesteps.device
    half_dim = dim // 2
    embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
    embeddings = timesteps[:, None] * embeddings[None, :]
    embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
    return embeddings

