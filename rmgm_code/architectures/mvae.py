import torch
import torch.nn as nn

class ImageEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(ImageEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.image_features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 16, 256),
            nn.SiLU(),
        )
        self.mean = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)

        def forward(self, x):
            h = self.image_features(x)
            return self.mean(h), self.logvar(h)


class TrajectoryEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(TrajectoryEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.trajectory_features = nn.Sequential(
            nn.Linear(200, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
        )
        self.mean = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)

        def forward(self, x):
            h = self.trajectory_features(x)
            return self.mean(h), self.logvar(h)


class ProductOfExperts(nn.Module):
    def forward(self, mean, logvar, eps=1e-8):
        std = torch.exp(logvar/2)
        