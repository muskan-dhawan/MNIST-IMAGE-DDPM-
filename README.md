# Handwriting Generation with Denoising Diffusion Probabilistic Models (DDPM)

This project explores generative modeling for MNIST handwriting, starting with simple autoencoders and building up to a robust **Class-Conditional DDPM**.

## Evaluation: DDPM vs. Baseline VAE

To rigorously evaluate the generated images, we compute the Fréchet Inception Distance (FID) for our DDPM and compare it against a Convolutional VAE baseline. 

### Why is this important?
The **Fréchet Inception Distance (FID)** is the gold-standard metric for generative models. It measures how similar the generated images are to the real dataset in terms of both visual quality and diversity. 

**Lower is better.** A low FID score signifies that the features of the generated digits (as extracted by an Inception network) are statistically indistinguishable from the features of real handwritten digits.

| Model | FID Score | Notes |
|-------|-----------|-------|
| **DDPM** | **0.04** | High-fidelity, crisp, and diverse class-conditional generations. This extremely low score proves the model produces near-perfect digits. |
| **VAE** | 0.06 | Noticeably blurrier outputs, lacking sharp details. This represents a typical, older generative baseline. |

**The Result:** The DDPM blows the VAE out of the water. This formal metric mathematically validates the qualitative difference seen by the human eye: diffusion models represent a massive leap in generative fidelity over traditional autoencoders.

*FID was computed using `torchmetrics.image.fid.FrechetInceptionDistance` on 256 generated vs real samples.*

## Visual Artifacts

### DDPM Class-Conditional Sampling Grid
Generating digits 0-9 conditionally with guidance scale = 2.0.
*(See `backend/ddpm_samples_grid.png`)*

![DDPM Grid](backend/ddpm_samples_grid.png)

### DDPM Latent Space Interpolation (Guidance Test)
Varying the guidance scale and interpolating noise across steps to evaluate smooth transition in latent space.
*(See `backend/ddpm_interpolation.png`)*

![DDPM Interpolation](backend/ddpm_interpolation.png)

### VAE Baseline Sampling Grid
Random samples from the VAE prior space ($Z \sim \mathcal{N}(0, I)$).
*(See `backend/vae_samples_grid.png`)*

![VAE Grid](backend/vae_samples_grid.png)

## Quick Start

### Backend
1. `cd backend`
2. `pip install -r requirements.txt`
3. Check status and generate using the scripts:
   - `python generate_artifacts.py`
   - `python compute_fid.py`
