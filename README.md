# 🎨 Full-Stack Conditional DDPM for MNIST Digit Generation

An end-to-end full-stack artificial intelligence system demonstrating **Conditional Denoising Diffusion Probabilistic Models (DDPM)** capable of high-fidelity digit synthesis from explicit numerical conditioning (classes 0–9). 

This project bridges empirical academic generative training in **TensorFlow/Keras** with high-performance production inference in **PyTorch**, coupled with automated deployment to **Hugging Face Spaces**.

## ✨ Architecture & Core Highlights
- **UNet Denoising Backbone:** Incorporates multi-resolution encoder-decoder layers, residual connection blocks (`ResBlock`), and dense conditioning embedding projections.
- **Classifier-Free Guidance (CFG):** Trains simultaneously on target digit labels and an explicit `NULL_CLASS = 10` representation to allow dynamic inference guidance scaling ($G_{scale} \ge 2.0$) for enhanced visual sharpness.
- **Cosine Beta Scheduling:** Replaces standard linear beta schedules with smooth trigonometric variance progression ($s=0.008$) to prevent extreme degradation in early forward diffusion timesteps.

## 🗃️ Codebase Topography
- `DDPM_MNIST.ipynb` — Interactive research and training notebook containing the complete mathematical formulation, custom forward noise injection, conditioning loops, and multi-digit sampling visualizers.
- `convert_weights.py` — Interoperability bridge that parses saved Keras models (`.weights.h5` via `h5py`), transposes weight tensor dimensions from TF format `(H, W, C_in, C_out)` into PyTorch format `(C_out, C_in, H, W)`, and exports production PyTorch dictionaries (`.pt`).
- `inspect_h5.py` — Diagnostic script for traversing and verifying layer architectures inside HDF5 neural network weight binaries.
- `deploy_to_hf.py` — Automated CI/CD script utilizing `huggingface_hub` to package, sanitize, and push inference backends directly to Hugging Face Spaces (`shivv01/mnist-backend`).
- `backend/` & `frontend/` — decoupled REST service architectures for presenting an intuitive web UI to end-users.

## ⚙️ Installation & Workflow Guidance

### 1. Environment Setup
Create a dedicated Virtual Environment and install cross-platform AI libraries:
```bash
git clone https://github.com/shivvrai/mnist_image_ddpm.git
cd mnist_image_ddpm

# Install core machine learning & deployment dependencies
pip install torch torchvision tensorflow tensorflow_datasets h5py huggingface_hub tqdm matplotlib numpy
```

### 2. Weight Transformation (TensorFlow ➡️ PyTorch)
After training in Keras or modifying `ddpm_mnist_cond_best.weights.h5`, run the weight conversion bridge before serving inference:
```bash
python convert_weights.py
```
*Outputs compiled PyTorch weight files directly into `backend/weights/ddpm_unet.pt` and `backend/weights/mnist_classifier.pt`.*

### 3. Deploying to Hugging Face Spaces
To automate publishing your latest backend iterations to the cloud without manual git-lfs setups:
```bash
# Ensure you have your Hugging Face API access token ready
python deploy_to_hf.py
```
