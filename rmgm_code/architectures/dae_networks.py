import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, latent_dim, first_layer_dim):
        super(Encoder, self).__init__()
        self.fc = nn.Linear(first_layer_dim, 512)
        self.feature_extractor = nn.Sequential(
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, latent_dim)
        )

    def set_first_layer(self, layer_dim):
        self.fc = nn.Linear(layer_dim, 512)
    
    def forward(self, x):
        h = self.fc(x)
        return self.feature_extractor(h)
    
    
class Decoder(nn.Module):
    def __init__(self, latent_dim, last_layer_dim):
        super(Decoder, self).__init__()
        self.feature_reconstructor = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 512),
            nn.SiLU(),
        )
        self.fc = nn.Linear(512, last_layer_dim)

    def set_last_layer(self, layer_dim):
        self.fc = nn.Linear(512, layer_dim)

    def forward(self, z):
        x_hat = self.feature_reconstructor(z)
        return self.fc(x_hat)

