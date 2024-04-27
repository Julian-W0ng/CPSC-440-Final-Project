import torch
import torch.nn as nn
import torch.nn.functional as F

def rsample(z_mean, z_log_sig, num_samples):

    b, k = z_mean.shape
    log_sig_q = z_log_sig.unsqueeze(1).expand(-1, num_samples, -1)
    mu_q = z_mean.unsqueeze(1).expand(-1, num_samples, -1)

    z = torch.randn(b, num_samples, k).to(mu_q.device)
    
    # make mu_q, and log_sig_q [B, N, K]
    mu_q = mu_q.unsqueeze(1).expand(-1, num_samples, -1)
    log_sig_q = log_sig_q.unsqueeze(1).unsqueeze(2).expand(-1, num_samples, k)

    x_q = mu_q + z * torch.exp(log_sig_q)

    return x_q

#adapted from: https://colab.research.google.com/drive/19b3Lq8woSXIBPo6qfeKzDL4r9rHKWsqm?usp=sharing&source=post_page-----d8e0cfc2245b--------------------------------#scrollTo=iFnL4o5u-Tib
class VAE_Spectogram(nn.Module):
    def __init__(self, latent_dim=2, height=128, width=128, channels=1):
        super(VAE_Spectogram, self).__init__()
        self.latent_dim = latent_dim

        # Encoder layers
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

        #Decoder layers
        self.t_conv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1)
        self.t_conv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1)
        self.t_conv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1)
        self.t_conv4 = nn.ConvTranspose2d(64, channels, kernel_size=3, padding=1)

        self.flatten_size = width//8 * height//8 * 512
        # Latent vectors mu and sigma
        self.fc1 = nn.Linear(self.flatten_size, latent_dim)
        self.fc2 = nn.Linear(self.flatten_size, latent_dim)

        # Sampling vector
        self.fc3 = nn.Linear(latent_dim, self.flatten_size)
        

    def forward(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = rsample(z_mean, z_log_var, 1)
        x_reconst = self.decoder(z)
        return x_reconst

    def encoder(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, self.flatten_size)  # Flatten the tensor
        z_mean = self.fc1(x)
        z_log_var = self.fc2(x)
        return z_mean, z_log_var
    
    def decoder(self, z):
        x = F.relu(self.fc3(z))
        x = x.view(-1, 512, 7, 7)
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = F.relu(self.t_conv3(x))
        x = torch.sigmoid(self.t_conv4(x))
        return x
    
    def elbo(self, x, num_samples=1):
        z_mean, z_log_var = self.encoder(x)
        z = rsample(z_mean, z_log_var, num_samples)
        x_reconst = self.decoder(z)
        reconst_loss = F.mse_loss(x_reconst, x, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
        return reconst_loss + kl_div