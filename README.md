### MNIST Image Generation using DDPM
An end-to-end generative AI project that implements a Denoising Diffusion Probabilistic Model (DDPM) to generate handwritten digits from the MNIST dataset. The project features a trained diffusion model, a Python backend API for serving the model, and an interactive frontend web application.

Live Demo: mnist-image-ddpm.vercel.app

🚀 Project Architecture
The repository is organized into three main components:

Model & Training (DDPM_MNIST.ipynb): A Jupyter Notebook containing the step-by-step implementation of the DDPM framework, including the forward diffusion process (adding noise), the backward denoising process, and the training loop using a custom U-Net architecture.

Backend (/backend): A lightweight Python API service that loads the trained weights (saved as .h5 or .pth), processes requests, and handles the iterative reverse diffusion sampling to generate digits.

Frontend (/frontend): A modern, responsive web interface built with JavaScript and CSS that connects to the backend API, allowing users to trigger image generation and visualize the diffusion process in real-time.

📂 Repository Structure
Plaintext
├── backend/               # API server to serve the generative model
├── frontend/              # Web UI assets (HTML, CSS, JS) for the user interface
├── .gitignore             # Git ignore file
├── DDPM_MNIST.ipynb       # Jupyter Notebook for DDPM training and experimentation
└── inspect_h5.py          # Helper utility to inspect model weights and HDF5 structures
🛠️ Technical Overview
Denoising Diffusion Probabilistic Models (DDPM)
DDPMs are a class of generative models that learn to generate data by reversing a gradual noisy process.

Forward Process: The model systematically adds Gaussian noise to a clean MNIST digit over a series of time steps T until it becomes pure noise.

Reverse Process: A neural network (typically a U-Net with residual blocks and attention mechanisms) is trained to predict the noise added at each step. By subtracting the predicted noise, the model can start from pure random noise and iteratively reconstruct a clean, crisp digit image.

💻 Getting Started
Prerequisites
Python 3.8+

Jupyter Notebook (to run the training script)

Setup & Installation
Clone the repository:

Bash
git clone https://github.com/muskan-dhawan/MNIST-IMAGE-DDPM-.git
cd MNIST-IMAGE-DDPM-
Model Training & Inspection:

Open DDPM_MNIST.ipynb in your Jupyter environment to review or re-run the training process.

If you need to inspect the contents or architecture of your saved .h5 model weights file, run the utility script:

Bash
python inspect_h5.py
Running the Backend:

Navigate to the backend directory, install the required dependencies (e.g., torch, tensorflow, numpy), and start the server:

Bash
cd backend
# Install your specific requirements, for example:
# pip install -r requirements.txt
python app.py
Running the Frontend:

Open the frontend/index.html file directly in your browser, or serve it using a local development server (e.g., Live Server in VS Code).

🌐 Deployment
Frontend: Deployed and hosted globally on Vercel at mnist-image-ddpm.vercel.app.

🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page if you want to contribute to the project.

📝 License
This project is open-source. Please check the repository settings or contact the author for specific licensing details.
