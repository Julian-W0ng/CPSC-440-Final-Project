import torch
from torch.nn import functional as F
from torch import nn

class VariationalTransformerEncoder(nn.Module):
    def __init__(self, device, nheads=1, sequence_length=5*44100, channels=1, dropout=0.1, ff_dim=10):
        super(VariationalTransformerEncoder, self).__init__()
        assert channels % nheads == 0, "channels must be divisible by nheads"
        # self.pos_embedding = nn.Embedding(sequence_length, channels, device=device)
        self.transformerLayers = nn.TransformerEncoderLayer(d_model=channels, nhead=nheads, dim_feedforward=ff_dim, batch_first=True, dropout=dropout, device=device)
        self.transformerEncoder = nn.TransformerEncoder(self.transformerLayers, num_layers=1)
        # TODO: fully connected layer to output mu and sigma
        # The latent space does not need to match the input space
        latent_dim = sequence_length*channels
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_log_sigma = nn.Linear(latent_dim, latent_dim)

    def forward(self, x):   
        # B x T x C
        # x = x + self.pos_embedding(torch.arange(x.size(1)).unsqueeze(0).to(x.device))
        x = self.transformerEncoder(x)
        x = x.view(x.shape[0], -1)
        mu = self.fc_mu(x)
        sigma = torch.exp(self.fc_log_sigma(x))
        return mu, sigma
    
class VariationalTransformerDecoder(nn.Module):
    def __init__(self, device, nheads=5, sequence_length=5*44100, channels=1, dropout=0.1, ff_dim=5):
        super(VariationalTransformerDecoder, self).__init__()
        #dont need to use sequence length for the embedding since we are using the latent space
        # self.pos_embedding = nn.Embedding(sequence_length, channels, device=device)
        # TODO: figure out how to use the transformer decoder and add the the sampling of the latent space

        self.transformerLayers = nn.TransformerDecoderLayer(d_model=channels, nhead=nheads, dim_feedforward=ff_dim, batch_first=True, dropout=dropout, device=device)
        self.transformerDecoder = nn.TransformerDecoder(self.transformerLayers, num_layers=1)

    def forward(self, x, z):
        mask = torch.triu(torch.ones(x.size(1), x.size(1)), diagonal=1).bool().to(x.device)
        x = self.transformerDecoder(tgt=x, memory=z, tgt_mask=mask)
        return x
        
def sample_latent(mu, sigma):
    # B x T x C
    # Sample from the normal distribution
    # mu is the mean and sigma is the standard deviation
    # We need to sample from the normal distribution to get the latent space
    return torch.normal(mu, sigma).unsqueeze(-1)


def elbo_loss(x, x_hat, mu, sigma):
    # Reconstruction Loss
    recon_loss = nn.functional.mse_loss(x_hat, x)

    # KL Divergence
    kl_div = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
    return recon_loss + kl_div

def sample_wav(decoder, device, sample_length, sample_size, channels):
    decoder.eval()
    z = torch.zeros(sample_size, sample_length, channels).to(device)
    output = torch.randn(sample_size, sample_length, channels).to(device)
    for i in range(sample_length):
        output = decoder(output, z)
    return output

