import torch.nn as nn


class ImageEncoder(nn.Module):
    def __init__(self, latent_dimension):
        super(ImageEncoder, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.SiLU(),
        )

        self.fc_mean = nn.Linear(128 * 7 * 7, latent_dimension)
        self.fc_logvar = nn.Linear(128 * 7 * 7, latent_dimension)

    def set_latent_dim(self, latent_dim):
        self.fc_mean = nn.Linear(128 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(128 * 7 * 7, latent_dim)

    def forward(self, x):
        h = self.feature_extractor(x)
        h = h.view(h.size(0), -1)
        return self.fc_mean(h), self.fc_logvar(h)
        

class ImageDecoder(nn.Module):
    def __init__(self, latent_dimension):
        super(ImageDecoder, self).__init__()
        self.latent_fc = nn.Linear(latent_dimension, 128 * 7 * 7)
        self.feature_reconstructor = nn.Sequential(
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
        )

    def set_latent_dim(self, latent_dim):
        self.latent_fc = nn.Linear(latent_dim, 512)

    def forward(self, z):
        x_hat = self.feature_reconstructor(self.latent_fc(z))
        return x_hat.view(x_hat.size(0), 128, 7, 7)


class TrajectoryEncoder(nn.Module):
    def __init__(self, latent_dimension):
        super(TrajectoryEncoder, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(200, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
        )

        self.fc_mean = nn.Linear(256, latent_dimension)
        self.fc_logvar = nn.Linear(256, latent_dimension)

    def set_latent_dim(self, latent_dim):
        self.fc_mean = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
    
    def forward(self, x):
        h = self.feature_extractor(x)

        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h)
        return mean, logvar


class TrajectoryDecoder(nn.Module):
    def __init__(self, latent_dimension):
        super(TrajectoryDecoder, self).__init__()
        self.latent_fc = nn.Linear(latent_dimension, 256)
        self.feature_reconstructor = nn.Sequential(
            nn.SiLU(),
            nn.Linear(256, 512),
            nn.SiLU(),
            nn.Linear(512, 200)
        )

    def set_latent_dim(self, latent_dim):
        self.latent_fc = nn.Linear(latent_dim, 256)

    def forward(self, z):
        return self.feature_reconstructor(self.latent_fc(z))
    
    