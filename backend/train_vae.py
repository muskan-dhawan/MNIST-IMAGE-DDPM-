import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from vae import VAE

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train_vae(epochs=15, batch_size=128, lr=1e-3, latent_dim=20, save_path="weights/vae_mnist.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training VAE on {device} for {epochs} epochs")
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model = VAE(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    model.train()
    for epoch in range(1, epochs + 1):
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
        print(f"Epoch {epoch}/{epochs}, Loss: {train_loss / len(train_loader.dataset):.4f}")
        
    torch.save(model.state_dict(), save_path)
    print(f"Saved VAE weights to {save_path}")

if __name__ == "__main__":
    train_vae()
