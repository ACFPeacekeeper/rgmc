import torch.nn as nn


class CommonEncoder(nn.Module):
    def __init__(self, latent_dimension):
        super(CommonEncoder, self).__init__()
        self.latent_dimension = latent_dimension
        self.latent_fc = nn.Linear(latent_dimension, latent_dimension)

    def set_latent_dim(self, latent_dim):
        self.latent_fc = nn.Linear(latent_dim, latent_dim)
        self.latent_dimension = latent_dim

    def forward(self, x):
        return self.latent_fc(x)


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
    

class MNISTEncoder(nn.Module):
    def __init__(self, latent_dimension):
        super(MNISTEncoder, self).__init__()
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
      

class SVHNEncoder(nn.Module):
    def __init__(self, latent_dimension):
        super(SVHNEncoder, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.GELU(),
            nn.Conv2d(32, 32 * 2, 4, 2, 1),
            nn.GELU(),
            nn.Conv2d(32 * 2, 32 * 4, 4, 2, 1),
            nn.GELU(),
        )

        self.latent_fc = nn.Linear(32 * 32 * 2, latent_dimension)

    def set_latent_dim(self, latent_dim):
        self.latent_fc = nn.Linear(32 * 32 * 2, latent_dim)

    def forward(self, x):
        h = self.feature_extractor(x)
        h = h.view(h.size(0), -1)
        return self.latent_fc(h)


class MNISTDecoder(nn.Module):
    def __init__(self, latent_dimension):
        super(MNISTDecoder, self).__init__()
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


class SVHNDecoder(nn.Module):
    def __init__(self, latent_dimension):
        super(SVHNDecoder, self).__init__()
        self.latent_fc = nn.Linear(latent_dimension, 32 * 32 * 2)
        self.feature_reconstructor = nn.Sequential(
            nn.GELU(),
            nn.ConvTranspose2d(32 * 4, 32 * 2, 4, 2, 1),
            nn.GELU(),
            nn.ConvTranspose2d(32 * 2, 32, 4, 2, 1),
            nn.GELU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
        )

    def set_latent_dim(self, latent_dim):
        self.latent_fc = nn.Linear(latent_dim, 32 * 32 * 2)

    def forward(self, z):
        x_hat = self.latent_fc(z)
        x_hat = x_hat.view(x_hat.size(0), 128, 4, 4)
        return self.feature_reconstructor(x_hat)