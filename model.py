import torch
from torch import nn

class VariatinoalTransformerEncoder(nn.Module):
    def __init__(self, device, nheads=5, sequence_length=5*44100, channels=1, dropout=0.1, ff_dim=2048):
        super(VariatinoalTransformerEncoder, self).__init__()
        self.pos_embedding = nn.Embedding(sequence_length, channels, device=device)
        self.transformerLayers = nn.TransformerEncoderLayer(d_model=channels, nhead=nheads, dim_feedforward=ff_dim, batch_first=True, dropout=dropout, device=device)
        self.transformerEncoder = nn.TransformerEncoder(self.transformerLayers, num_layers=1)
        # TODO: fully connected layer to output mu and sigma
        # The latent space does not need to match the input space

    def forward(self, x):
        pass
        # Return mu and sigma

class VariationalTransformerDecoder(nn.Module):
    def __init__(self, device, nheads=5, sequence_length=5*44100, channels=1, dropout=0.1, ff_dim=2048):
        super(VariationalTransformerDecoder, self).__init__()
        self.pos_embedding = nn.Embedding(sequence_length, channels, device=device)
        # TODO: figure out how to use the transformer decoder and add the the sampling of the latent space
        # self.transformerLayers = nn.TransformerDecoderLayer(d_model=channels, nhead=nheads, dim_feedforward=ff_dim, batch_first=True, dropout=dropout, device=device)
        # self.transformerDecoder = nn.TransformerDecoder(self.transformerLayers, num_layers=1)

    def forward(self, x):
        pass
        # Return B x T x C
        # This will be used to generate the output waveform and calculate the loss