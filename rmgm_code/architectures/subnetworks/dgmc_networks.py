import torch
import torch.nn as nn
import torch.nn.functional as F


class MHDCommonEncoder(nn.Module):
    def __init__(self, common_dim, latent_dimension):
        super(MHDCommonEncoder, self).__init__()
        self.common_dim = common_dim
        self.latent_dimension = latent_dimension
        self.feature_extractor = nn.Sequential(
            nn.Linear(common_dim, 512),
            Swish(),
            nn.Linear(512, 512),
            Swish(),
        )
        self.latent_fc = nn.Linear(512, latent_dimension)

    def set_latent_dim(self, latent_dim):
        self.latent_fc = nn.Linear(512, latent_dim)
        self.latent_dimension = latent_dim

    def forward(self, x):
        return F.normalize(self.latent_fc(self.feature_extractor(x)), dim=-1)


class MHDCommonDecoder(nn.Module):
    def __init__(self, common_dim, latent_dimension):
        super(MHDCommonDecoder, self).__init__()
        self.common_dim = common_dim
        self.latent_dimension = latent_dimension
        self.latent_fc = nn.Linear(latent_dimension, 512)
        self.feature_reconstructor = nn.Sequential(
            Swish(),
            nn.Linear(512, 512),
            Swish(),
            nn.Linear(512, common_dim)
        )

    def set_latent_dim(self, latent_dim):
        self.latent_fc = nn.Linear(512, latent_dim)
        self.latent_dimension = latent_dim

    def forward(self, x):
        x_hat = self.latent_fc(x)
        return self.feature_reconstructor(x_hat)
    

class MHDImageProcessor(nn.Module):
    def __init__(self, common_dim):
        super(MHDImageProcessor, self).__init__()
        self.common_dim = common_dim
        self.image_features = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            Swish(),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            Swish(),
        )
        self.projector = nn.Linear(128 * 7 * 7, common_dim)

    def forward(self, x):
        h = self.image_features(x)
        h = h.view(h.size(0), -1)
        return self.projector(h)


class MHDImageDecoder(nn.Module):
    def __init__(self, common_dim):
        super(MHDImageDecoder, self).__init__()
        self.common_dim = common_dim
        self.projector = nn.Linear(common_dim, 128 * 7 * 7)
        self.image_reconstructor = nn.Sequential(
            Swish(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            Swish(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
        )

    def forward(self, z):
        x_hat = self.projector(z)
        return self.image_reconstructor(x_hat)


class MHDSoundProcessor(nn.Module):
    def __init__(self, common_dim):
        super(MHDSoundProcessor, self).__init__()
        self.common_dim = common_dim
        self.sound_features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(1, 128), stride=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.projector = nn.Linear(2048, common_dim)

    def forward(self, x):
        h = self.sound_features(x)
        h = h.view(h.size(0), -1)
        return self.projector(h)


class MHDTrajectoryProcessor(nn.Module):
    def __init__(self, common_dim):
        super(MHDTrajectoryProcessor, self).__init__()
        self.common_dim = common_dim
        self.trajectory_features = nn.Sequential(
            nn.Linear(200, 512),
            Swish(),
            nn.Linear(512, 512),
            Swish(),
        )
        self.projector = nn.Linear(512, common_dim)

    def forward(self, x):
        h = self.trajectory_features(x)
        return self.projector(h)
    

class MHDTrajectoryDecoder(nn.Module):
    def __init__(self, common_dim):
        super(MHDTrajectoryDecoder, self).__init__()
        self.common_dim = common_dim
        self.projector = nn.Linear(common_dim, 512)
        self.trajectory_reconstructor = nn.Sequential(
            Swish(),
            nn.Linear(512, 512),
            Swish(),
            nn.Linear(512, 200)
        )

    def forward(self, z):
        x_hat = self.projector(z)
        return self.trajectory_reconstructor(x_hat)        


class MHDLabelProcessor(nn.Module):
    def __init__(self, common_dim):
        super(MHDLabelProcessor, self).__init__()
        self.common_dim = common_dim
        self.projector = nn.Linear(10, common_dim)

    def forward(self, x):
        return self.projector(x)


class MHDLabelDecoder(nn.Module):
    def __init__(self, common_dim):
        self.common_dim = common_dim
        self.projector = nn.Linear(common_dim, 10)

    def forward(self, z):
        return self.projector(z)


class MHDJointProcessor(nn.Module):
    def __init__(self, common_dim):
        super(MHDJointProcessor, self).__init__()
        self.common_dim = common_dim

        self.img_features = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            Swish(),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            Swish(),
        )

        self.trajectory_features = nn.Sequential(
            nn.Linear(200, 512),
            Swish(),
            nn.Linear(512, 512),
            Swish(),
        )

        self.projector = nn.Linear(128 * 7 * 7 + 512, common_dim)

    def forward(self, x):
        x_img, x_trajectory = x['image'], x['trajectory']

        # Image
        h_img = self.img_features(x_img)
        h_img = h_img.view(h_img.size(0), -1)

        # Trajectory
        h_trajectory = self.trajectory_features(x_trajectory)

        return self.projector(torch.cat((h_img, h_trajectory), dim=-1))


class MHDJointDecoder(nn.Module):
    def __init__(self, common_dim):
        super(MHDJointDecoder, self).__init__()
        self.common_dim = common_dim
        self.projector = nn.Linear(common_dim, 128 * 7 * 7 + 512)
        self.image_reconstructor = nn.Sequential(
            Swish(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            Swish(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
        )

        self.trajectory_reconstructor = nn.Sequential(
            Swish(),
            nn.Linear(512, 512),
            Swish(),
            nn.Linear(512, 200)
        )

    def forward(self, z):
        x_hat = self.projector(z)

        # Image recon
        img_hat = self.image_reconstructor(x_hat)
        img_hat = img_hat.view(img_hat.size(0), 128, 7, 7)

        # Trajectory recon
        traj_hat = self.trajectory_reconstructor(x_hat)

        return {"image": img_hat, "trajectory": traj_hat}

"""


Extra components


"""

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
