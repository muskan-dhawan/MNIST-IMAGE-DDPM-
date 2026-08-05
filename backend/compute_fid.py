import argparse
import math
import os
import time
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

from ddpm import UNet, gen_samples, gen_samples_ddim
from vae import VAE

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("WARNING: CUDA not available. Using CPU.")
    return device

def load_models(device):
    print("Loading DDPM UNet...")
    ddpm_model = UNet().to(device)
    ddpm_path = "weights/ddpm_unet.pt"
    if not os.path.exists(ddpm_path):
        ddpm_path = os.path.join(os.path.dirname(__file__), "weights/ddpm_unet.pt")
    ddpm_model.load_state_dict(torch.load(ddpm_path, map_location=device, weights_only=True))
    ddpm_model.eval()

    print("Loading Class-Conditional VAE baseline...")
    vae_model = VAE(latent_dim=32, num_classes=10).to(device)
    vae_path = "weights/vae_mnist.pt"
    if not os.path.exists(vae_path):
        vae_path = os.path.join(os.path.dirname(__file__), "weights/vae_mnist.pt")
    vae_model.load_state_dict(torch.load(vae_path, map_location=device, weights_only=True))
    vae_model.eval()

    return ddpm_model, vae_model

def generate_preview_grid(ddpm_model, vae_model, device, save_path="eval_preview_10_samples.png", guidance_scale=3.0):
    print("\n=== Generating 10-Image Verification Grid ===")
    ddpm_model.eval()
    vae_model.eval()

    # 1. Collect 1 real image for each digit 0-9 from test set
    print("Fetching 10 real test-set images (digits 0-9)...")
    dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    real_images = {}
    for img, label in dataset:
        if label not in real_images:
            real_images[label] = img.numpy()[0]  # 28x28
        if len(real_images) == 10:
            break
    real_row = [real_images[i] for i in range(10)]

    # 2. Generate 10 DDPM samples (digits 0-9)
    print(f"Generating 10 DDPM samples (digits 0-9) with guidance scale = {guidance_scale}...")
    cond = list(range(10))
    with torch.no_grad():
        ddpm_imgs_tensor = gen_samples(ddpm_model, n_samples=10, conditioning=cond, guidance_scale=guidance_scale, seed=42)
    ddpm_row = [ddpm_imgs_tensor[i, 0].cpu().numpy() for i in range(10)]

    # 3. Generate 10 cVAE samples from prior (class conditioned 0-9)
    print("Generating 10 class-conditional VAE samples (digits 0-9)...")
    with torch.no_grad():
        torch.manual_seed(42)
        z = torch.randn(10, vae_model.latent_dim, device=device)
        vae_imgs_tensor = vae_model.decode(z, labels=list(range(10))).cpu()
    vae_row = [vae_imgs_tensor[i, 0].numpy() for i in range(10)]

    # 4. Build visualization grid (3 rows x 10 columns)
    cell_sz = 28
    scale = 4  # Upscale to 112x112 per digit for high visual clarity
    up_sz = cell_sz * scale
    pad = 4
    label_w = 160

    total_w = label_w + (up_sz + pad) * 10 + pad
    total_h = (up_sz + pad) * 3 + pad

    grid_img = Image.new("RGB", (total_w, total_h), (30, 30, 30))
    draw = ImageDraw.Draw(grid_img)

    row_names = ["Real Test Set", f"DDPM (GS={guidance_scale})", "cVAE (Baseline)"]
    rows_data = [real_row, ddpm_row, vae_row]

    for r_idx, (r_name, r_data) in enumerate(zip(row_names, rows_data)):
        y_pos = pad + r_idx * (up_sz + pad)
        # Draw row title
        draw.text((10, y_pos + up_sz // 2 - 10), r_name, fill=(255, 255, 255))
        
        # Paste 10 images
        for c_idx, img_arr in enumerate(r_data):
            x_pos = label_w + pad + c_idx * (up_sz + pad)
            img_uint8 = np.clip(img_arr * 255.0, 0, 255).astype(np.uint8)
            pil_digit = Image.fromarray(img_uint8, mode="L").convert("RGB")
            pil_digit = pil_digit.resize((up_sz, up_sz), Image.NEAREST)
            grid_img.paste(pil_digit, (x_pos, y_pos))

    grid_img.save(save_path)
    print(f"\n[SUCCESS] Verification grid saved to {save_path}")

def load_real_test_data(num_samples=10000, batch_size=256):
    print(f"Loading {num_samples} real images from MNIST test set...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),       # 3 channels RGB for Inception
        transforms.Lambda(lambda x: (x * 255).clamp(0, 255).byte()) # uint8 [0, 255]
    ])
    dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    real_images = []
    for data, _ in loader:
        real_images.append(data)
        if sum([x.size(0) for x in real_images]) >= num_samples:
            break
            
    real_images = torch.cat(real_images, dim=0)[:num_samples]
    return real_images

@torch.no_grad()
def generate_ddpm_dataset(model, num_samples=10000, batch_size=200, guidance_scale=3.0, device='cuda', force_regenerate=False):
    cache_path = f"ddpm_samples_{num_samples}_gs{guidance_scale}.pt"
    if not force_regenerate and os.path.exists(cache_path):
        print(f"Loading cached DDPM samples from {cache_path}...")
        return torch.load(cache_path, weights_only=True)[:num_samples]
    
    print(f"Generating {num_samples} balanced DDPM samples (1,000 per digit 0-9) on {device}...")
    model.eval()
    samples = []
    
    # Balanced classes: repeat [0, 1, ..., 9] exactly num_samples/10 times
    all_cond = np.tile(np.arange(10), math.ceil(num_samples / 10))[:num_samples]
    
    start_time = time.time()
    for i in tqdm(range(0, num_samples, batch_size), desc="DDPM Sampling"):
        bs = min(batch_size, num_samples - i)
        cond_slice = all_cond[i : i + bs]
        
        x = gen_samples_ddim(model, n_samples=bs, conditioning=cond_slice, guidance_scale=guidance_scale, ddim_steps=50)
        # Convert to 3 channels and uint8 [0, 255]
        x = x.repeat(1, 3, 1, 1)
        x = (x * 255.0).clamp(0, 255).byte()
        samples.append(x.cpu())
        
    elapsed = time.time() - start_time
    print(f"DDPM generation completed in {elapsed:.1f}s ({num_samples/elapsed:.1f} imgs/sec)")
    res = torch.cat(samples, dim=0)[:num_samples]
    torch.save(res, f"ddpm_samples_{num_samples}_gs{guidance_scale}.pt")
    return res

@torch.no_grad()
def generate_vae_dataset(model, num_samples=10000, batch_size=500, device='cuda', force_regenerate=False):
    cache_path = f"cvae_samples_{num_samples}.pt"
    if not force_regenerate and os.path.exists(cache_path):
        print(f"Loading cached Class-Conditional VAE samples from {cache_path}...")
        return torch.load(cache_path, weights_only=True)[:num_samples]
    
    print(f"Generating {num_samples} class-balanced cVAE samples (1,000 per digit 0-9) on {device}...")
    model.eval()
    samples = []
    
    # Balanced classes: repeat [0, 1, ..., 9] exactly num_samples/10 times
    all_cond = np.tile(np.arange(10), math.ceil(num_samples / 10))[:num_samples]
    
    start_time = time.time()
    for i in tqdm(range(0, num_samples, batch_size), desc="cVAE Sampling"):
        bs = min(batch_size, num_samples - i)
        cond_slice = all_cond[i : i + bs]
        z = torch.randn(bs, model.latent_dim, device=device)
        x = model.decode(z, labels=cond_slice)
        
        # Convert to 3 channels and uint8 [0, 255]
        x = x.repeat(1, 3, 1, 1)
        x = (x * 255.0).clamp(0, 255).byte()
        samples.append(x.cpu())
        
    elapsed = time.time() - start_time
    print(f"cVAE generation completed in {elapsed:.1f}s ({num_samples/elapsed:.1f} imgs/sec)")
    res = torch.cat(samples, dim=0)[:num_samples]
    torch.save(res, f"cvae_samples_{num_samples}.pt")
    return res

def compute_fid_evaluation(num_samples=10000, batch_size=200, guidance_scale=3.0, force_regenerate=False):
    device = get_device()
    ddpm_model, vae_model = load_models(device)

    # 1. Load Real Data
    real_images = load_real_test_data(num_samples=num_samples)
    
    # 2. Generate Model Datasets
    ddpm_images = generate_ddpm_dataset(ddpm_model, num_samples=num_samples, batch_size=batch_size, guidance_scale=guidance_scale, device=device, force_regenerate=force_regenerate)
    vae_images = generate_vae_dataset(vae_model, num_samples=num_samples, batch_size=500, device=device, force_regenerate=force_regenerate)

    # Free U-Net and VAE VRAM before loading InceptionV3
    print("Deallocating U-Net and VAE from VRAM to make room for InceptionV3...")
    del ddpm_model, vae_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3. Compute FID for DDPM vs Real
    print("\nComputing Frechet Inception Distance (FID) with standard feature=2048...")
    eval_batch_size = 64
    
    print(">>> Running InceptionV3 for DDPM vs Real...")
    fid_ddpm = FrechetInceptionDistance(feature=2048).to(device)
    for i in tqdm(range(0, num_samples, eval_batch_size), desc="DDPM FID Update"):
        fid_ddpm.update(real_images[i:i+eval_batch_size].to(device), real=True)
        fid_ddpm.update(ddpm_images[i:i+eval_batch_size].to(device), real=False)
    ddpm_score = fid_ddpm.compute().item()
    print(f"\n===========================================")
    print(f"* Final DDPM FID Score (10k, GS={guidance_scale}): {ddpm_score:.4f}")
    print(f"===========================================")

    # 4. Compute FID for VAE vs Real
    print(">>> Running InceptionV3 for VAE vs Real...")
    fid_vae = FrechetInceptionDistance(feature=2048).to(device)
    for i in tqdm(range(0, num_samples, eval_batch_size), desc="VAE FID Update"):
        fid_vae.update(real_images[i:i+eval_batch_size].to(device), real=True)
        fid_vae.update(vae_images[i:i+eval_batch_size].to(device), real=False)
    vae_score = fid_vae.compute().item()
    print(f"===========================================")
    print(f"* Final VAE Baseline FID Score (10k): {vae_score:.4f}")
    print(f"===========================================\n")

    # 5. Save Results
    results_path = "fid_scores_10k.txt"
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("MNIST Digit Generation Evaluation (10,000 samples vs Test Set)\n")
        f.write(f"Feature space: InceptionV3 (feature=2048)\n")
        f.write(f"DDPM Guidance Scale: {guidance_scale}\n\n")
        f.write(f"DDPM FID Score: {ddpm_score:.4f}\n")
        f.write(f"Class-Conditional VAE FID: {vae_score:.4f}\n")
    print(f"Saved formal evaluation scores to {results_path}")

    # 6. Generate Top 30 Generations Grid
    from generate_artifacts import generate_top_30_grid
    generate_top_30_grid(device=device, guidance_scale=guidance_scale)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDPM vs VAE FID Evaluation on MNIST")
    parser.add_argument("--preview-only", action="store_true", help="Generate 10-image preview verification grid only")
    parser.add_argument("--eval-10k", action="store_true", help="Run full 10,000 samples FID evaluation")
    parser.add_argument("--guidance-scale", type=float, default=3.0, help="Classifier-free guidance scale for DDPM")
    parser.add_argument("--batch-size", type=int, default=200, help="Batch size for generating DDPM samples")
    parser.add_argument("--force-regenerate", action="store_true", help="Force regenerate 10,000 samples instead of using cache")
    args = parser.parse_args()

    device = get_device()

    if args.preview_only or (not args.eval_10k and not args.preview_only):
        ddpm_model, vae_model = load_models(device)
        generate_preview_grid(ddpm_model, vae_model, device, save_path="eval_preview_10_samples.png", guidance_scale=args.guidance_scale)

    if args.eval_10k:
        compute_fid_evaluation(num_samples=10000, batch_size=args.batch_size, guidance_scale=args.guidance_scale, force_regenerate=args.force_regenerate)
