import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, latent_dim, first_layer_dim):
        super(Encoder, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(first_layer_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, latent_dim)
        )
    
    def forward(self, x):
        return self.feature_extractor(x)
    
    
class Decoder(nn.Module):
    def __init__(self, latent_dim, last_layer_dim):
        super(Decoder, self).__init__()
        self.feature_reconstructor = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 512),
            nn.SiLU(),
            nn.Linear(512, last_layer_dim)
        )

    def forward(self, z):
        return self.feature_reconstructor(z)

