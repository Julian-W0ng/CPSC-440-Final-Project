import torch
from torch.nn import functional as F
from torch import nn
import numpy as np


def kl_q_p_exact(params_q: torch.Tensor, params_p: torch.Tensor) -> torch.Tensor:
  """
  Ground truth function used for unit test. Do not modify.
  @param params_q: [B, K+1]: parameters of B many q distribution, the first K elements are the mean, the last element is the log standard deviation
  @param params_p: [B, K+1]: parameters of B many p distribution, the first K elements are the mean, the last element is the log standard deviation
  @return kl_q_p_e [B]: KL(q||p) for q||p
  """
  # Init 
  b, k_ = params_q.shape
  k = k_ - 1
  mu_q, log_sig_q = params_q[:, :-1], params_q[:, -1]  # [B, K], [B]
  mu_p, log_sig_p = params_p[:, :-1], params_p[:, -1]  # [B, K], [B]

  kl = 0.5 * (k * torch.exp(2 * (log_sig_q - log_sig_p)) + ((mu_q - mu_p)**2).sum(dim=1) * torch.exp(-2 * log_sig_p) - k + 2 * k * (log_sig_p - log_sig_q))
  return kl

def log_prob(x: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
  """
  Ground truth function used for unit test. Do not modify.
  @param x:          [B, N, K]: input data (observations of Gaussian distribution), there are N samples for each of the B batches, each sample has K dimensions.
  @param mu:         [B, N, K]: mean of Gaussian distribution.
  @param log_sigma:  [B, N]: log of standard deviation of Gaussian distribution.
  @return log_prob:  [B, N]: log Gaussian probability for each sample in the batch.
  """
  B, N, K = x.shape
  log_prob = torch.zeros(B, N).to(x.device)
  log_prob = -K * log_sigma - ((x - mu)**2).sum(dim=-1) / (2 * torch.exp(2 * log_sigma)) - K / 2 * np.log(2 * np.pi)

  return log_prob

def elbo_loss(x, x_hat, mu, log_sigma):
    # Reconstruction Loss
    recon_loss = nn.functional.mse_loss(x_hat, x, reduction='sum')

    # KL Divergence
    kl_div = -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp())
    return recon_loss + kl_div

def rsample(params_q: torch.Tensor, num_samples: int) -> torch.Tensor:
    """
    Ground truth implementation of rsample. Do not modify.
    @param params_q:        [B, K+1]: parameters of B many q distribution, the first K elements are the mean, the last element is the log standard deviation
    @param num_samples:     int: number of samples to create
    @return x_q:            [B, N, K]: samples from q distribution, where N is the number of samples
    """
    # Init
    b, k_ = params_q.shape
    k = k_ - 1
    mu_q, log_sig_q = params_q[:, :-1], params_q[:, -1]  # [B, K], [B]

    z = torch.randn(b, num_samples, k).to(mu_q.device)
    
    # make mu_q, and log_sig_q [B, N, K]
    mu_q = mu_q.unsqueeze(1).expand(-1, num_samples, -1)
    log_sig_q = log_sig_q.unsqueeze(1).unsqueeze(2).expand(-1, num_samples, k)

    x_q = mu_q + z * torch.exp(log_sig_q)

    return x_q
class ExpLayer(nn.Module):
    def forward(self, x):
        return torch.exp(x)
    
class SimpleVAE(nn.Module):
  """
  Simple VAE with convolutional encoder and decoder.
  """
  def __init__(self, K=2, num_filters=32, sequence_length=5*44100, channels=1, sample_rate=44100):
    """
    We aim to build a simple VAE with the following architecture for the MNIST dataset.
    @param K: int: Bottleneck dimensionality
    @param num_filters: int: Number of filters [default: 32]
    """

    super(SimpleVAE, self).__init__()

    self.kernel_size = sample_rate//2 + 1
    self.image_size = sequence_length*channels
    self.size_after_conv = (self.image_size)

    self.flat_size_after_conv = num_filters * self.size_after_conv

    # encoder
    self.encoder = nn.Sequential(
        nn.Conv1d(1, num_filters, self.kernel_size, padding_mode='zeros', padding=self.kernel_size//2),
        nn.ReLU(),
        nn.Conv1d(num_filters, num_filters, self.kernel_size,  padding_mode='zeros', padding=self.kernel_size//2),
        nn.ReLU(),
        nn.Conv1d(num_filters, num_filters, self.kernel_size, padding_mode='zeros', padding=self.kernel_size//2),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(self.flat_size_after_conv, K+1)
    )
    
    # decoder
    self.decoder = nn.Sequential(
        nn.Linear(K, num_filters * self.size_after_conv),
        nn.Unflatten(1, (num_filters, self.size_after_conv)),
        nn.ReLU(),
        nn.ConvTranspose1d(num_filters, num_filters, self.kernel_size, padding_mode='zeros', padding=self.kernel_size//2),
        nn.ReLU(),
        nn.ConvTranspose1d(num_filters, num_filters, self.kernel_size, padding_mode='zeros', padding=self.kernel_size//2),
        nn.ReLU(),
        nn.ConvTranspose1d(num_filters, 1, self.kernel_size, padding_mode='zeros', padding=self.kernel_size//2),
        nn.Tanh()
    )

    # decoder variance parameter: for simplicity, we define a scalar log_sig_x for all pixels
    self.log_sig_x = nn.Parameter(torch.zeros(()))

  def decode(self, samples_z):
    """
    Wrapper for decoder
    @param samples_z:   [B, N, K]: samples from the latent space, B: batch size, N: number of samples, K: latent dimensionality
    @return mu_xs:      [B, N, C, H, W]: mean of the reconstructed data, B: batch size, N: number of samples, D: data dimensionality
    """
    b, n, k = samples_z.shape
    s_z = samples_z.reshape(b * n, -1)                                     # [B*N, K]
    s_z = self.decoder(s_z)                                             # [B*N, D]
    mu_xs = s_z.reshape(b, n, 1, self.image_size)         # [B, N, C, H, W]
    return mu_xs

  def elbo(self, x, n=1):
    """
    Run input end to end through the VAE and compute the ELBO using n samples of z
    @param x:       [B, C, H, W]: input image, B: batch size, C: number of channels, H: height, W: width
    @param n:       int: number of samples of z sample and reconstruction samples
    @return elbo:   scalar: aggregated ELBO loss for each image in the batch
    """
    phi = self.encoder(x)     # [B, K+1] <- [B, C, W]
    zs = rsample(phi, n)      # [B, N, K] <- [B, K+1]
    mu_xs = self.decode(zs)   # [B, N, C, W] <- [B, N, K]

    # b, c, w = x.shape
    # x_flat = x.view(b, 1, -1).expand(-1, n, -1)                 # [B, N, C*W] <- [B, 1, C*W] <- [B, C, H, W]
    # mu_xs_flat = mu_xs.view(b, n, -1)                           # [B, N, C*W]
    # log_sig_x = self.log_sig_x.view(1, 1).expand(x.size(0), n)  # [B, N]

    # note: we use the exact KL divergence here, but we could also use the Monte Carlo approximation
    # note: we didn't use the ELBO loss implemented in Q1.5, because it is less numerically stable
    # elbo_loss = log_prob(x_flat, mu_xs_flat, log_sig_x).mean() - kl_q_p_exact(phi, torch.zeros_like(phi)).mean()
        # Reconstruction Loss
    mu_xs = mu_xs.squeeze(2)

    recon_loss = nn.functional.mse_loss(mu_xs, x, reduction='sum')

    # KL Divergence
    kl_div = kl_q_p_exact(phi, torch.zeros_like(phi)).mean()
    # print(recon_loss, kl_div)
    return recon_loss + kl_div


    return elbo_loss
  
  def forward(self, x, n=1):

    out = self.encoder(x)
    zs = rsample(out, n)
    mu_xs = self.decode(zs)

    return mu_xs
