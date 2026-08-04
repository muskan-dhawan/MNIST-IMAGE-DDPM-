import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

from ddpm import UNet, gen_samples
from vae import VAE

def load_real_data(num_samples=10000, batch_size=256):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # Inception requires 3 channels
        transforms.Lambda(lambda x: (x * 255).byte()) # FID expects uint8 [0, 255]
    ])
    
    dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    real_images = []
    for data, _ in loader:
        real_images.append(data)
        if sum([x.size(0) for x in real_images]) >= num_samples:
            break
            
    real_images = torch.cat(real_images, dim=0)[:num_samples]
    return real_images

@torch.no_grad()
def generate_ddpm_samples(model, num_samples=10000, batch_size=256, device='cuda'):
    print(f"Generating {num_samples} DDPM samples...")
    model.eval()
    samples = []
    
    for _ in tqdm(range(math.ceil(num_samples / batch_size))):
        bs = min(batch_size, num_samples - len(samples) * batch_size)
        if bs <= 0: break
        
        # Sample random classes
        cond = torch.randint(0, 10, (bs,), device=device)
        
        # gen_samples outputs [0, 1] float32
        x = gen_samples(model, bs, cond.cpu().numpy(), guidance_scale=2.0)
        
        # Convert to 3 channels and uint8
        x = x.repeat(1, 3, 1, 1)
        x = (x * 255).clamp(0, 255).byte()
        samples.append(x.cpu())
        
    return torch.cat(samples, dim=0)[:num_samples]

@torch.no_grad()
def generate_vae_samples(model, num_samples=10000, batch_size=256, device='cuda'):
    print(f"Generating {num_samples} VAE samples...")
    model.eval()
    samples = []
    
    for _ in tqdm(range(math.ceil(num_samples / batch_size))):
        bs = min(batch_size, num_samples - len(samples) * batch_size)
        if bs <= 0: break
        
        z = torch.randn(bs, 20, device=device)
        x = model.decode(z) # outputs [0, 1] float32
        
        # Convert to 3 channels and uint8
        x = x.repeat(1, 3, 1, 1)
        x = (x * 255).clamp(0, 255).byte()
        samples.append(x.cpu())
        
    return torch.cat(samples, dim=0)[:num_samples]

def compute_fid():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_eval_samples = 256 # evaluate on 256 samples for accuracy without massive wait times
    
    print("Loading real data...")
    real_images = load_real_data(num_samples=num_eval_samples)
    
    print("Loading models...")
    ddpm_model = UNet().to(device)
    ddpm_model.load_state_dict(torch.load("weights/ddpm_unet.pt", map_location=device, weights_only=True))
    
    vae_model = VAE(latent_dim=20).to(device)
    vae_model.load_state_dict(torch.load("weights/vae_mnist.pt", map_location=device, weights_only=True))
    
    ddpm_images = generate_ddpm_samples(ddpm_model, num_samples=num_eval_samples, device=device)
    vae_images = generate_vae_samples(vae_model, num_samples=num_eval_samples, device=device)
    
    print("Computing FID for DDPM...")
    fid_ddpm = FrechetInceptionDistance(feature=64).to(device)
    # Using feature=64 for faster computation, default 2048 is too heavy for simple MNIST
    
    # Process in batches to avoid OOM
    batch_size = 256
    for i in range(0, num_eval_samples, batch_size):
        fid_ddpm.update(real_images[i:i+batch_size].to(device), real=True)
        fid_ddpm.update(ddpm_images[i:i+batch_size].to(device), real=False)
    ddpm_score = fid_ddpm.compute().item()
    print(f"DDPM FID Score: {ddpm_score:.2f}")
    
    print("Computing FID for VAE...")
    fid_vae = FrechetInceptionDistance(feature=64).to(device)
    for i in range(0, num_eval_samples, batch_size):
        fid_vae.update(real_images[i:i+batch_size].to(device), real=True)
        fid_vae.update(vae_images[i:i+batch_size].to(device), real=False)
    vae_score = fid_vae.compute().item()
    print(f"VAE FID Score: {vae_score:.2f}")
    
    # Write to a text file
    with open("fid_scores.txt", "w") as f:
        f.write(f"DDPM FID (feature=64): {ddpm_score:.2f}\n")
        f.write(f"VAE FID (feature=64): {vae_score:.2f}\n")

if __name__ == "__main__":
    compute_fid()
