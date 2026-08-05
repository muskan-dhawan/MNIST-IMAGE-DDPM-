import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, latent_dim=32, num_classes=10):
        super(VAE, self).__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        
        # Encoder: receives image (1 channel) + one-hot class map (10 channels) = 11 channels
        self.enc_conv1 = nn.Conv2d(1 + num_classes, 32, 3, stride=2, padding=1)
        self.enc_conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_var = nn.Linear(64 * 7 * 7, latent_dim)
        
        # Decoder: receives latent vector z (latent_dim) + one-hot class vector (num_classes)
        self.fc_dec = nn.Linear(latent_dim + num_classes, 64 * 7 * 7)
        self.dec_conv1 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1)
        
    def _one_hot(self, labels, device, batch_size):
        if labels is None:
            # Fallback to zeros if no label provided
            return torch.zeros(batch_size, self.num_classes, device=device)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long, device=device)
        labels = labels.to(device).long()
        return F.one_hot(labels, num_classes=self.num_classes).float()

    def encode(self, x, labels):
        batch_size = x.size(0)
        device = x.device
        one_hot = self._one_hot(labels, device, batch_size)
        # Expand one-hot vector to spatial feature map (B, num_classes, 28, 28)
        spatial_cond = one_hot.view(batch_size, self.num_classes, 1, 1).expand(-1, -1, x.size(2), x.size(3))
        x_cond = torch.cat([x, spatial_cond], dim=1)
        
        h = F.relu(self.enc_conv1(x_cond))
        h = F.relu(self.enc_conv2(h))
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_var(h)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z, labels):
        batch_size = z.size(0)
        device = z.device
        one_hot = self._one_hot(labels, device, batch_size)
        z_cond = torch.cat([z, one_hot], dim=1)
        
        h = F.relu(self.fc_dec(z_cond))
        h = h.view(h.size(0), 64, 7, 7)
        h = F.relu(self.dec_conv1(h))
        return torch.sigmoid(self.dec_conv2(h))
        
    def forward(self, x, labels=None):
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, labels), mu, logvar
