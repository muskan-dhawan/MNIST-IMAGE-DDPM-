import os
import torch
import numpy as np
from torchvision.utils import make_grid
from PIL import Image, ImageDraw

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
    print("Generating Class-Conditional VAE samples grid...")
    model.eval()
    samples = []
    
    # 10 rows (classes 0-9), 10 columns (samples per class)
    for c in range(10):
        cond = torch.full((10,), c, dtype=torch.long, device=device)
        z = torch.randn(10, model.latent_dim, device=device)
        with torch.no_grad():
            x = model.decode(z, labels=cond)
            samples.append(x.cpu())
            
    samples = torch.cat(samples, dim=0)
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
            
            x = _p_sample_step(x, eps, _betas[step], _alphas[step], _alpha_hats[step], step)
            
        x = (x + 1.0) * 0.5
        
    grid = make_grid(x.cpu(), nrow=10, padding=2, normalize=True)
    img = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(img).save("ddpm_interpolation.png")
    print("Saved ddpm_interpolation.png")

def generate_top_30_grid(device, guidance_scale=3.0):
    print("\n=== Generating Top 30 DDPM Generations (Ranked by Classifier Confidence) ===")
    samples_path = f"ddpm_samples_10000_gs{guidance_scale}.pt"
    if not os.path.exists(samples_path):
        alt_path = os.path.join(os.path.dirname(__file__), samples_path)
        if os.path.exists(alt_path):
            samples_path = alt_path
        else:
            print(f"[WARNING] Could not find {samples_path}. Please run compute_fid.py --eval-10k first.")
            return

    print(f"Loading cached 10,000 samples from {samples_path}...")
    samples = torch.load(samples_path, weights_only=True)  # (10000, 3, 28, 28) uint8

    # Load MNIST Classifier to evaluate sample quality
    from classifier import MNISTClassifier
    classifier = MNISTClassifier().to(device)
    clf_path = "weights/mnist_classifier.pt"
    if not os.path.exists(clf_path):
        clf_path = os.path.join(os.path.dirname(__file__), clf_path)
    classifier.load_state_dict(torch.load(clf_path, map_location=device, weights_only=True))
    classifier.eval()

    print("Scoring generated samples using PyTorch MNIST Classifier on GPU...")
    all_probs = []
    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i+batch_size, 0:1, :, :].to(device).float() / 255.0  # Grayscale [0, 1]
            probs = classifier(batch)
            all_probs.append(probs.cpu())
    all_probs = torch.cat(all_probs, dim=0)  # (10000, 10)

    # For each digit 0-9, select top 3 highest confidence samples
    top_samples = {c: [] for c in range(10)}
    for c in range(10):
        # Samples for class c were generated at indices c, c+10, c+20, ...
        indices = [idx for idx in range(c, len(samples), 10)]
        confidences = [(all_probs[idx, c].item(), idx) for idx in indices]
        confidences.sort(key=lambda x: x[0], reverse=True)
        top_3_indices = [idx for _, idx in confidences[:3]]
        top_samples[c] = [samples[idx, 0].numpy() for idx in top_3_indices]  # 28x28

    # Build visual representation (3 rows x 10 columns, upscaled 4x)
    cell_sz = 28
    scale = 4  # Upscale to 112x112 per digit for crisp presentation
    up_sz = cell_sz * scale
    pad = 6
    label_w = 180

    total_w = label_w + (up_sz + pad) * 10 + pad
    total_h = (up_sz + pad) * 3 + pad

    grid_img = Image.new("RGB", (total_w, total_h), (25, 25, 25))
    draw = ImageDraw.Draw(grid_img)

    row_names = ["Rank #1 Quality", "Rank #2 Quality", "Rank #3 Quality"]
    for r_idx, r_name in enumerate(row_names):
        y_pos = pad + r_idx * (up_sz + pad)
        draw.text((15, y_pos + up_sz // 2 - 8), r_name, fill=(240, 240, 240))
        
        for c_idx in range(10):
            x_pos = label_w + pad + c_idx * (up_sz + pad)
            img_uint8 = top_samples[c_idx][r_idx]
            pil_digit = Image.fromarray(img_uint8, mode="L").convert("RGB")
            pil_digit = pil_digit.resize((up_sz, up_sz), Image.NEAREST)
            grid_img.paste(pil_digit, (x_pos, y_pos))

    save_path = "top_30_ddpm_generations.png"
    grid_img.save(save_path)
    print(f"[SUCCESS] Top 30 generated samples grid saved to {save_path}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ddpm_model = UNet().to(device)
    ddpm_path = "weights/ddpm_unet.pt"
    if not os.path.exists(ddpm_path):
        ddpm_path = os.path.join(os.path.dirname(__file__), ddpm_path)
    ddpm_model.load_state_dict(torch.load(ddpm_path, map_location=device, weights_only=True))
    
    vae_model = VAE(latent_dim=32, num_classes=10).to(device)
    vae_path = "weights/vae_mnist.pt"
    if not os.path.exists(vae_path):
        vae_path = os.path.join(os.path.dirname(__file__), vae_path)
    vae_model.load_state_dict(torch.load(vae_path, map_location=device, weights_only=True))
    
    generate_ddpm_grid(ddpm_model, device)
    generate_vae_grid(vae_model, device)
    generate_ddpm_interpolation(ddpm_model, device)
    generate_top_30_grid(device, guidance_scale=3.0)
