import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, latent_dimension, first_layer_dim):
        super(Encoder, self).__init__()
        self.fc = nn.Linear(first_layer_dim, 256)
        self.feature_extractor = nn.Sequential(
            nn.GELU(),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
        )

        self.fc_mean = nn.Linear(256, latent_dimension)
        self.fc_logvar = nn.Linear(256, latent_dimension)

    def set_first_layer(self, layer_dim):
        self.fc = nn.Linear(layer_dim, 512)

    def set_latent_dim(self, latent_dim):
        self.fc_mean = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
    
    def forward(self, x):
        h = self.fc(x)
        h = self.feature_extractor(h)

        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h)
        return mean, logvar
    
    
class Decoder(nn.Module):
    def __init__(self, latent_dimension, last_layer_dim):
        super(Decoder, self).__init__()
        self.latent_fc = nn.Linear(latent_dimension, 256)
        self.feature_reconstructor = nn.Sequential(
            nn.GELU(),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU()
        )
        self.fc = nn.Linear(256, last_layer_dim)

    def set_last_layer(self, layer_dim):
        self.fc = nn.Linear(512, layer_dim)

    def set_latent_dim(self, latent_dim):
        self.latent_fc = nn.Linear(latent_dim, 256)

    def forward(self, z):
        x_hat = self.feature_reconstructor(self.latent_fc(z))
        return self.fc(x_hat)