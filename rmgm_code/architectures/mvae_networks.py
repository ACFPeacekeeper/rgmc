import torch
import torch.nn as nn

class ImageEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(ImageEncoder, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 512),
            nn.SiLU(),
        )

        self.fc_mean = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

    def forward(self, x):
        h = self.feature_extractor(x)
        return self.fc_mean(h), self.fc_logvar(h)
        
class ImageDecoder(nn.Module):
    def __init__(self, latent_dim):
        super(ImageDecoder, self).__init__()
        self.feature_reconstructor = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 128 * 7 * 7),
            nn.Unflatten(dim=1, unflattened_size=(128, 7, 7)),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
        )

    def forward(self, z):
        return self.feature_reconstructor(z)

class TrajectoryEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(TrajectoryEncoder, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(200, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
        )

        self.fc_mean = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
    
    def forward(self, x):
        h = self.feature_extractor(x)

        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h)
        return mean, logvar

class TrajectoryDecoder(nn.Module):
    def __init__(self, latent_dim):
        super(TrajectoryDecoder, self).__init__()
        self.feature_reconstructor = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 512),
            nn.SiLU(),
            nn.Linear(512, 200)
        )

    def forward(self, z):
        return self.feature_reconstructor(z)
    
    