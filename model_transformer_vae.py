import torch
from torch.nn import functional as F
from torch import nn

class VariationalTransformerEncoder(nn.Module):
    def __init__(self, device, nheads=5, sequence_length=5*44100, channels=1, dropout=0.1, ff_dim=5):
        super(VariationalTransformerEncoder, self).__init__()
        assert channels % nheads == 0, "channels must be divisible by nheads"
        self.transformerLayers = nn.TransformerEncoderLayer(d_model=channels, nhead=nheads, dim_feedforward=ff_dim, batch_first=True, dropout=dropout, device=device)
        self.transformerEncoder = nn.TransformerEncoder(self.transformerLayers, num_layers=1)
        latent_dim = sequence_length*channels
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_log_sigma = nn.Linear(latent_dim, latent_dim)

    def forward(self, x):   
        # B x T x C
        x = self.transformerEncoder(x)
        x = x.view(x.shape[0], -1)
        mu = self.fc_mu(x)
        log_sigma = self.fc_log_sigma(x)
        return mu, log_sigma
    
class VariationalTransformerDecoder(nn.Module):
    def __init__(self, device, nheads=5, sequence_length=5*44100, channels=1, dropout=0.1, ff_dim=10):
        super(VariationalTransformerDecoder, self).__init__()
        #dont need to use sequence length for the embedding since we are using the latent space
        # self.pos_embedding = nn.Embedding(sequence_length, channels, device=device)
        # TODO: figure out how to use the transformer decoder and add the the sampling of the latent space
        self.transformerLayers = nn.TransformerDecoderLayer(d_model=channels, nhead=nheads, dim_feedforward=ff_dim, batch_first=True, dropout=dropout, device=device)
        self.transformerDecoder = nn.TransformerDecoder(self.transformerLayers, num_layers=1)
        self.sequence_length = sequence_length

    def forward(self, x, z):
        mask = torch.triu(torch.ones(x.shape[1], x.shape[1]), diagonal=1).bool().unsqueeze(0).repeat(x.shape[0], 1, 1).to(x.device)
        x = self.transformerDecoder(tgt=x, memory=z, tgt_mask=mask)
        return x

class VariationalTransformerAutoencoder(nn.Module):
    def __init__(self, device, nheads=5, sequence_length=5*44100, channels=1, dropout=0.1, ff_dim=10):
        super(VariationalTransformerAutoencoder, self).__init__()
        self.encoder = VariationalTransformerEncoder(device, nheads, sequence_length, channels, dropout, ff_dim)
        self.decoder = VariationalTransformerDecoder(device, nheads, sequence_length, channels, dropout, ff_dim)
        self.sequence_length = sequence_length

    def forward(self, x):
        mu, log_sigma = self.encoder(x)
        z = self.reparameterize(mu, log_sigma)
        x_hat = self.decoder(x, z)
        return x_hat, mu, log_sigma
        
    def reparameterize(self, mu, log_sigma):
        epsilon = torch.randn_like(mu).to(mu.device)
        sigma = torch.exp(0.5 * log_sigma)
        z = mu + sigma * epsilon
        return z.unsqueeze(-1)
class SimpleVAE(nn.Module):
  """
  Simple VAE with convolutional encoder and decoder.
  """
  def __init__(self, K=44100, num_filters=32, sequence_length=5*44100, channels=1, sample_rate=44100):
    """
    We aim to build a simple VAE with the following architecture for the MNIST dataset.
    @param K: int: Bottleneck dimensionality
    @param num_filters: int: Number of filters [default: 32]
    """

    super(SimpleVAE, self).__init__()

    self.kernel_size = sequence_length//5
    self.image_size = sequence_length
    self.size_after_conv = self.image_size - self.kernel_size + 1
    self.flat_size_after_conv = num_filters * self.size_after_conv

    # encoder
    self.encoder = nn.Sequential(
        nn.Conv1d(1, num_filters, self.kernel_size),
        nn.ReLU(),
        nn.Conv1d(num_filters, num_filters, self.kernel_size),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(self.flat_size_after_conv, K+1)
    )

    # decoder
    self.decoder = nn.Sequential(
        nn.Linear(K, self.flat_size_after_conv),
        nn.Unflatten(1, (num_filters, self.size_after_conv, self.size_after_conv)),
        nn.ReLU(),
        nn.ConvTranspose1d(num_filters, num_filters, self.kernel_size),
        nn.ReLU(),
        nn.ConvTranspose1d(num_filters, 1, self.kernel_size),
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
    s_z = samples_z.view(b * n, -1)                                     # [B*N, K]
    s_z = self.decoder(s_z)                                             # [B*N, D]
    mu_xs = s_z.view(b, n, 1, self.image_size, self.image_size)         # [B, N, C, H, W]
    return mu_xs

  def elbo(self, x, n=1):
    """
    Run input end to end through the VAE and compute the ELBO using n samples of z
    @param x:       [B, C, H, W]: input image, B: batch size, C: number of channels, H: height, W: width
    @param n:       int: number of samples of z sample and reconstruction samples
    @return elbo:   scalar: aggregated ELBO loss for each image in the batch
    """
    phi = self.encoder(x)     # [B, K+1] <- [B, C, H, W]
    zs = rsample(phi, n)      # [B, N, K] <- [B, K+1]
    mu_xs = self.decode(zs)   # [B, N, C, H, W] <- [B, N, K]

    b, c, h, w = x.shape
    x_flat = x.view(b, 1, -1).expand(-1, n, -1)                 # [B, N, C*H*W] <- [B, 1, C*H*W] <- [B, C, H, W]
    mu_xs_flat = mu_xs.view(b, n, -1)                           # [B, N, C*H*W]
    log_sig_x = self.log_sig_x.view(1, 1).expand(x.size(0), n)  # [B, N]

    # note: we use the exact KL divergence here, but we could also use the Monte Carlo approximation
    # note: we didn't use the ELBO loss implemented in Q1.5, because it is less numerically stable
    elbo_loss = log_prob(x_flat, mu_xs_flat, log_sig_x).mean() - kl_q_p_exact(phi, torch.zeros_like(phi)).mean()
    return elbo_loss
  

def gt_log_prob(x: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
  """
  Ground truth function used for unit test. Do not modify.
  @param x:          [B, N, K]: input data (observations of Gaussian distribution), there are N samples for each of the B batches, each sample has K dimensions.
  @param mu:         [B, N, K]: mean of Gaussian distribution.
  @param log_sigma:  [B, N]: log of standard deviation of Gaussian distribution.
  @return log_prob:  [B, N]: log Gaussian probability for each sample in the batch.
  """
  gt_log_prob = torch.distributions.normal.Normal(mu, log_sigma[:, :, None].exp()).log_prob(x)
  gt_log_prob = gt_log_prob.sum(dim=-1)
  return gt_log_prob

def elbo_loss(x, x_hat, mu, log_sigma):
    # Reconstruction Loss
    recon_loss = nn.functional.mse_loss(x_hat, x, reduction='sum')

    # KL Divergence
    kl_div = -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp())
    return recon_loss + kl_div

def sample(model: VariationalTransformerAutoencoder, device, sample_size, sequence_length, channels):
    model.eval()
    with torch.no_grad():
        # Start with a sequence containing just a start-of-sequence token
        z = torch.randn(sample_size, sequence_length, channels).to(device)
        x_hat = torch.zeros(sample_size, sequence_length, channels).to(device)
        for i in range(1, sequence_length):
            # Expand the sequence one token at a time
            x_hat[:, :i, :] = model.decoder(x_hat[:, :i, :], z)
    return x_hat 