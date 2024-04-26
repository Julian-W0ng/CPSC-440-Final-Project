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