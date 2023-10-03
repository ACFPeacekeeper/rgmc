import torch.nn as nn


class ImageEncoder(nn.Module):
    def __init__(self, latent_dimension):
        super(ImageEncoder, self).__init__()
        self.image_features = nn.Sequential(
            nn.Conv2d(2, 32, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.SiLU(),
        )

        self.fc_mean = nn.Linear(14400, latent_dimension)
        self.fc_logvar = nn.Linear(14400, latent_dimension)

    def set_latent_dim(self, latent_dim):
        self.fc_mean = nn.Linear(14400, latent_dim)
        self.fc_logvar = nn.Linear(14400, latent_dim)

    def forward(self, x):
        h = self.image_features(x)
        h = h.view(h.size(0), -1)
        return self.fc_mean(h), self.fc_logvar(h)
        

class ImageDecoder(nn.Module):
    def __init__(self, latent_dimension):
        super(ImageDecoder, self).__init__()
        self.latent_fc = nn.Linear(latent_dimension, 14400)
        self.feature_reconstructor = nn.Sequential(
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 2, 4, 2, 1),
        )

    def set_latent_dim(self, latent_dim):
        self.latent_fc = nn.Linear(latent_dim, 14400)

    def forward(self, z):
        x_hat = self.latent_fc(z)
        x_hat = x_hat.view(x_hat.size(0), 14400)
        return self.feature_reconstructor(x_hat)


class SoundEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(SoundEncoder, self).__init__()
        self.n_stack = 2
        self.sound_channels = 3
        self.sound_length = 2
        self.unrolled_sound_input = (
            self.n_stack * self.sound_channels * self.sound_length
        )

        self.snd_features = nn.Sequential(
            nn.Linear(self.unrolled_sound_input, 50),
            nn.SiLU(),
            nn.Linear(50, 50),
            nn.SiLU(),
        )
        self.projector = nn.Linear(50, latent_dim)

    def set_latent_dim(self, latent_dim):
        self.projector = nn.Linear(50, latent_dim)

    def forward(self, x):
        x = x.view(-1, self.unrolled_sound_input)
        h = self.snd_features(x)
        return self.projector(h)


class SoundDecoder(nn.Module):
    def __init__(self, latent_dimension):
        super(SoundDecoder, self).__init__()
        self.n_stack = 2
        self.sound_channels = 3
        self.sound_length = 2
        self.unrolled_sound_input = (
            self.n_stack * self.sound_channels * self.sound_length
        )
        self.latent_fc = nn.Linear(latent_dimension, 50)
        self.feature_reconstructor = nn.Sequential(
            nn.SiLU(),
            nn.Linear(50, 50),
            nn.SiLU(),
            nn.Linear(50, self.unrolled_sound_input)
        )

    def set_latent_dim(self, latent_dim):
        self.latent_fc = nn.Linear(latent_dim, 256)

    def forward(self, z):
        return self.feature_reconstructor(self.latent_fc(z))