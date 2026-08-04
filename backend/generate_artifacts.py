import torch
import numpy as np
from torchvision.utils import make_grid
from PIL import Image

from ddpm import UNet, gen_samples
from vae import VAE

def slerp(val, low, high):
    """Spherical interpolation. val has a range of 0 to 1."""
    if val <= 0: return low
    if val >= 1: return high
    
    omega = np.arccos(np.clip(np.dot(low / np.linalg.norm(low), high / np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0 - val) * low + val * high # L'Hopital's rule/LERP
    return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high

def generate_ddpm_grid(model, device):
    print("Generating DDPM class-conditional grid...")
    model.eval()
    samples = []
    
    # 10 rows (classes 0-9), 10 columns (samples per class)
    for c in range(10):
        cond = torch.full((10,), c, dtype=torch.long, device=device)
        # Generate outputs in [0, 1]
        x = gen_samples(model, 10, cond.cpu().numpy(), guidance_scale=2.0)
        samples.append(x.cpu())
        
    samples = torch.cat(samples, dim=0)
    grid = make_grid(samples, nrow=10, padding=2, normalize=True)
    
    img = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(img).save("ddpm_samples_grid.png")
    print("Saved ddpm_samples_grid.png")

def generate_vae_grid(model, device):
    print("Generating VAE samples grid...")
    model.eval()
    
    # Generate 100 random samples
    z = torch.randn(100, 20, device=device)
    with torch.no_grad():
        samples = model.decode(z).cpu()
        
    grid = make_grid(samples, nrow=10, padding=2, normalize=True)
    img = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(img).save("vae_samples_grid.png")
    print("Saved vae_samples_grid.png")

def generate_ddpm_interpolation(model, device):
    print("Generating DDPM latent interpolation...")
    model.eval()
    
    # Interpolate for digit '3'
    cond = torch.full((10,), 3, dtype=torch.long, device=device)
    null_cond = torch.full((10,), 10, dtype=torch.long, device=device)
    
    # We will interpolate between two random noise vectors
    z1 = np.random.randn(1, 1, 28, 28)
    z2 = np.random.randn(1, 1, 28, 28)
    
    z_interp = []
    for val in np.linspace(0, 1, 10):
        z = slerp(val, z1.flatten(), z2.flatten()).reshape(1, 1, 28, 28)
        z_interp.append(z)
        
    x = torch.tensor(np.concatenate(z_interp, axis=0), dtype=torch.float32, device=device)
    
    with torch.no_grad():
        # Denoise the interpolated noise
        for step in reversed(range(300)):
            t_batch = torch.full((10,), step, dtype=torch.long, device=device)
            
            # Since this relies on internal functions of DDPM not fully exposed, 
            # let's just do it cleanly using the model's math
            from ddpm import _betas, _alphas, _alpha_hats, _p_sample_step
            
            eps_u = model(x, t_batch, null_cond)
            eps_c = model(x, t_batch, cond)
            eps = eps_u + 2.0 * (eps_c - eps_u)
            
            x = _p_sample_step(x, eps, _betas[step], _alphas[step], _alpha_hats[step])
            
        x = (x + 1.0) * 0.5
        
    grid = make_grid(x.cpu(), nrow=10, padding=2, normalize=True)
    img = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(img).save("ddpm_interpolation.png")
    print("Saved ddpm_interpolation.png")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ddpm_model = UNet().to(device)
    ddpm_model.load_state_dict(torch.load("weights/ddpm_unet.pt", map_location=device, weights_only=True))
    
    vae_model = VAE(latent_dim=20).to(device)
    vae_model.load_state_dict(torch.load("weights/vae_mnist.pt", map_location=device, weights_only=True))
    
    generate_ddpm_grid(ddpm_model, device)
    generate_vae_grid(vae_model, device)
    generate_ddpm_interpolation(ddpm_model, device)
