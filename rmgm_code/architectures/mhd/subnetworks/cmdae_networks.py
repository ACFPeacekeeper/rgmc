import torch.nn as nn


class CommonEncoder(nn.Module):
    def __init__(self, latent_dimension):
        super(CommonEncoder, self).__init__()
        self.latent_dimension = latent_dimension
        self.fc_mean = nn.Linear(latent_dimension, latent_dimension)
        self.fc_logvar = nn.Linear(latent_dimension, latent_dimension)

    def set_latent_dim(self, latent_dim):
        self.fc_mean = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)
        self.latent_dimension = latent_dim

    def forward(self, x):
        return self.fc_mean(x), self.fc_logvar(x)
    
class CommonDecoder(nn.Module):
    def __init__(self, latent_dimension):
        super(CommonDecoder, self).__init__()
        self.latent_dimension = latent_dimension
        self.latent_fc = nn.Linear(latent_dimension, latent_dimension)

    def set_latent_dim(self, latent_dim):
        self.latent_fc = nn.Linear(latent_dim, latent_dim)
        self.latent_dimension = latent_dim

    def forward(self, z):
        return self.latent_fc(z)

class ImageEncoder(nn.Module):
    def __init__(self, latent_dimension):
        super(ImageEncoder, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.GELU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.GELU(),
        )
        self.latent_fc = nn.Linear(128 * 7 * 7, latent_dimension)

    def set_latent_dim(self, latent_dim):
        self.latent_fc = nn.Linear(128 * 7 * 7, latent_dim)

    def forward(self, x):
        h = self.feature_extractor(x)
        h = h.view(h.size(0), -1)
        return self.latent_fc(h)
        

class ImageDecoder(nn.Module):
    def __init__(self, latent_dimension):
        super(ImageDecoder, self).__init__()
        self.latent_fc = nn.Linear(latent_dimension, 128 * 7 * 7)
        self.feature_reconstructor = nn.Sequential(
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.GELU(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
        )

    def set_latent_dim(self, latent_dim):
        self.latent_fc = nn.Linear(latent_dim, 128 * 7 * 7)

    def forward(self, z):
        x_hat = self.latent_fc(z)
        x_hat = x_hat.view(x_hat.size(0), 128, 7, 7)
        return self.feature_reconstructor(x_hat)


class TrajectoryEncoder(nn.Module):
    def __init__(self, latent_dimension):
        super(TrajectoryEncoder, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(200, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
        )

        self.latent_fc = nn.Linear(256, latent_dimension)

    def set_latent_dim(self, latent_dim):
        self.fc_mean = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
    
    def forward(self, x):
        h = self.feature_extractor(x)
        return self.latent_fc(h)


class TrajectoryDecoder(nn.Module):
    def __init__(self, latent_dimension):
        super(TrajectoryDecoder, self).__init__()
        self.latent_fc = nn.Linear(latent_dimension, 256)
        self.feature_reconstructor = nn.Sequential(
            nn.GELU(),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, 200)
        )

    def set_latent_dim(self, latent_dim):
        self.latent_fc = nn.Linear(latent_dim, 256)

    def forward(self, z):
        return self.feature_reconstructor(self.latent_fc(z))