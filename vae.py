import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, input_size, latent_size, kernel_size, num_kernels, input_channels=1):
        super(VAE, self).__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels
        self.channels = input_channels
        self.stride = 2
        self.padding = kernel_size // 2 
        self.after_conv_size = input_size // self.stride
        self.hidden_size = num_kernels * self.after_conv_size
   
       
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, num_kernels, kernel_size, stride=self.stride, padding=self.padding),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU()
        )

        self.mu = nn.Linear(self.hidden_size, latent_size)
        self.logvar = nn.Linear(self.hidden_size, latent_size)

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Unflatten(1, (num_kernels, self.hidden_size // num_kernels)),
            nn.ConvTranspose1d(num_kernels, input_channels, kernel_size, stride=self.stride, padding=self.padding, output_padding=1),
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
        z = torch.randn(num_samples, self.latent_size).to(device)
        return self.decoder(z)

    def loss(self, x, x_recon, mu, logvar):
        recon_loss = nn.functional.mse_loss(x_recon, x, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_div