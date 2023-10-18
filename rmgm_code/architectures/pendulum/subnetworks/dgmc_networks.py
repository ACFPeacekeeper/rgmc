import torch
import torch.nn as nn
import torch.nn.functional as F


# Pendulum
class PendulumCommonEncoder(nn.Module):
    def __init__(self, common_dim, latent_dim):
        super(PendulumCommonEncoder, self).__init__()
        # Variables
        self.common_dim = common_dim
        self.latent_dim = latent_dim
        self.feature_extractor = nn.Sequential(nn.Linear(common_dim, 128), nn.GELU(), nn.Linear(128, latent_dim),)
    
    def set_latent_dim(self, latent_dim):
        self.feature_extractor = nn.Sequential(nn.Linear(self.common_dim, 128), nn.GELU(), nn.Linear(128, latent_dim),)
        self.latent_dim = latent_dim

    def set_common_dim(self, common_dim):
        self.feature_extractor = nn.Sequential(nn.Linear(common_dim, 128), nn.GELU(), nn.Linear(128, self.latent_dim),)
        self.common_dim = common_dim

    def forward(self, x):
        return F.normalize(self.feature_extractor(x), dim=-1)


class MHDCommonDecoder(nn.Module):
    def __init__(self, common_dim, latent_dimension):
        super(MHDCommonDecoder, self).__init__()
        self.common_dim = common_dim
        self.latent_dimension = latent_dimension
        self.latent_fc = nn.Linear(latent_dimension, 512)
        self.feature_reconstructor = nn.Sequential(
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
        )
        self.common_fc = nn.Linear(512, common_dim)

    def set_latent_dim(self, latent_dim):
        self.latent_fc = nn.Linear(512, latent_dim)
        self.latent_dimension = latent_dim

    def set_common_dim(self, common_dim):
        self.common_fc = nn.Linear(512, common_dim)
        self.common_dim = common_dim

    def forward(self, z):
        h = self.latent_fc(z)
        return self.common_fc(self.feature_reconstructor(h))
    

class PendulumImageProcessor(nn.Module):
    def __init__(self, common_dim):
        super(PendulumImageProcessor, self).__init__()
        self.common_dim = common_dim
        self.image_features = nn.Sequential(
            nn.Conv2d(2, 32, 4, 2, 1),
            nn.GELU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.GELU(),
        )
        self.projector = nn.Linear(14400, common_dim)

    def set_common_dim(self, common_dim):
        self.projector = nn.Linear(14400,  common_dim)
        self.common_dim = common_dim

    def forward(self, x):
        x = self.image_features(x)
        x = x.view(x.size(0), -1)
        return self.projector(x)


class MHDImageDecoder(nn.Module):
    def __init__(self, common_dim):
        super(MHDImageDecoder, self).__init__()
        self.common_dim = common_dim
        self.projector = nn.Linear(common_dim, 128 * 7 * 7)
        self.image_reconstructor = nn.Sequential(
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.GELU(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1)
        )

    def set_common_dim(self, common_dim):
        self.projector = nn.Linear(common_dim, 128 * 7 * 7)
        self.common_dim = common_dim

    def forward(self, z):
        x_hat = self.projector(z)
        x_hat = x_hat.view(x_hat.size(0), 128, 7, 7)
        return self.image_reconstructor(x_hat)


class PendulumSoundProcessor(nn.Module):
    def __init__(self, common_dim):
        super(PendulumSoundProcessor, self).__init__()
        self.common_dim = common_dim
        self.n_stack = 2
        self.sound_channels = 3
        self.sound_length = 2
        self.unrolled_sound_input = (
            self.n_stack * self.sound_channels * self.sound_length
        )

        self.snd_features = nn.Sequential(
            nn.Linear(self.unrolled_sound_input, 50),
            nn.GELU(),
            nn.Linear(50, 50),
            nn.GELU(),
        )
        self.projector = nn.Linear(50, common_dim)

    def set_common_dim(self, common_dim):
        self.projector = nn.Linear(50, common_dim)
        self.common_dim = common_dim

    def forward(self, x):
        x = x.view(-1, self.unrolled_sound_input)
        h = self.snd_features(x)
        return self.projector(h)


class MHDTrajectoryDecoder(nn.Module):
    def __init__(self, common_dim):
        super(MHDTrajectoryDecoder, self).__init__()
        self.common_dim = common_dim
        self.projector = nn.Linear(common_dim, 512)
        self.trajectory_reconstructor = nn.Sequential(
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 200)
        )

    def set_common_dim(self, common_dim):
        self.projector = nn.Linear(common_dim, 512)
        self.common_dim = common_dim

    def forward(self, h):
        x_hat = self.projector(h)
        return self.trajectory_reconstructor(x_hat)        


# Pendulum
class PendulumJointProcessor(nn.Module):
    def __init__(self, common_dim):
        super(PendulumJointProcessor, self).__init__()
        # Variables
        self.common_dim = common_dim
        self.n_stack = 2
        self.sound_channels = 3
        self.sound_length = 2
        self.unrolled_sound_input = (
            self.n_stack * self.sound_channels * self.sound_length
        )

        self.img_features = nn.Sequential(
            nn.Conv2d(2, 32, 4, 2, 1),
            nn.GELU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.GELU(),
        )

        self.snd_features = nn.Sequential(
            nn.Linear(self.unrolled_sound_input, 50),
            nn.GELU(),
            nn.Linear(50, 50),
            nn.GELU(),
        )
        self.projector = nn.Linear(14400 + 50, common_dim)

    def set_common_dim(self, common_dim):
        self.projector = nn.Linear(14400 + 50, common_dim)
        self.common_dim = common_dim

    def forward(self, x):

        x_img, x_snd = x[0], x[1]

        x_img = self.img_features(x_img)
        x_img = x_img.view(x_img.size(0), -1)

        x_snd = x_snd.view(-1, self.unrolled_sound_input)
        x_snd = self.snd_features(x_snd)
        return self.projector(torch.cat((x_img, x_snd), dim=-1))


class MHDJointDecoder(nn.Module):
    def __init__(self, common_dim):
        super(MHDJointDecoder, self).__init__()
        self.common_dim = common_dim
        self.projector = nn.Linear(common_dim, 128 * 7 * 7 + 512)
        self.image_reconstructor = nn.Sequential(
           nn.GELU(),
           nn.ConvTranspose2d(128, 64, 4, 2, 1),
           nn.GELU(),
           nn.ConvTranspose2d(64, 1, 4, 2, 1), 
        )

        self.trajectory_reconstructor = nn.Sequential(
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 200)
        )

    def set_common_dim(self, common_dim):
        self.projector = nn.Linear(common_dim, 128 * 7 * 7 + 512)
        self.common_dim = common_dim

    def forward(self, z):
        x_hat = self.projector(z)

        # Image recon
        img_hat = x_hat[:, :128 * 7 * 7]
        img_hat = img_hat.view(img_hat.size(0), 128, 7, 7)
        img_hat = self.image_reconstructor(img_hat)

        # Trajectory recon
        traj_hat = x_hat[:, 128 * 7 * 7:128 * 7 * 7 + 512]
        traj_hat = self.trajectory_reconstructor(traj_hat)

        return {"image": img_hat, "trajectory": traj_hat}