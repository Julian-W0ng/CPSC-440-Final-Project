import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, input_channels=1, num_hidden_layers=3):
        super(VAE, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.channels = input_channels
       
        self.encoder = nn.Sequential(
            # nn.BatchNorm1d(input_channels),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            *[nn.Linear(hidden_size, hidden_size), nn.ReLU()] * num_hidden_layers
        )

        self.mu = nn.Linear(hidden_size, latent_size)
        self.logvar = nn.Linear(hidden_size, latent_size)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            *[nn.Linear(hidden_size, hidden_size), nn.ReLU()] * num_hidden_layers,
            nn.Linear(self.hidden_size, input_size),
            nn.Tanh()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def sample(self, num_samples, device):
        z = torch.randn(num_samples, self.channels, self.latent_size).to(device)
        return self.decoder(z)

    def loss(self, x, x_recon, mu, logvar):
        recon_loss = nn.functional.mse_loss(x_recon, x, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_div