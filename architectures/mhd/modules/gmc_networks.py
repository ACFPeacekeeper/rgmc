import torch
import torch.nn as nn


# Code adapted from https://github.com/miguelsvasco/gmc
class MHDCommonEncoder(nn.Module):
    def __init__(self, common_dim, latent_dimension):
        super(MHDCommonEncoder, self).__init__()
        self.common_dim = common_dim
        self.latent_dimension = latent_dimension
        self.common_fc = nn.Linear(common_dim, 512)
        self.feature_extractor = nn.Sequential(
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
        )
        self.latent_fc = nn.Linear(512, latent_dimension)

    def set_latent_dim(self, latent_dim):
        self.latent_fc = nn.Linear(512, latent_dim)
        self.latent_dimension = latent_dim

    def set_common_dim(self, common_dim):
        self.common_fc = nn.Linear(common_dim, 512)
        self.common_dim = common_dim

    def forward(self, x):
        h = self.common_fc(x)
        return nn.functional.normalize(self.latent_fc(self.feature_extractor(h)), dim=-1)


class MHDImageProcessor(nn.Module):
    def __init__(self, common_dim):
        super(MHDImageProcessor, self).__init__()
        self.common_dim = common_dim
        self.image_features = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.GELU(),
        )
        self.projector = nn.Linear(128 * 7 * 7, common_dim)

    def set_common_dim(self, common_dim):
        self.projector = nn.Linear(128 * 7 * 7,  common_dim)
        self.common_dim = common_dim

    def forward(self, x):
        h = self.image_features(x)
        h = h.view(h.size(0), -1)
        return self.projector(h)


class MHDTrajectoryProcessor(nn.Module):
    def __init__(self, common_dim):
        super(MHDTrajectoryProcessor, self).__init__()
        self.common_dim = common_dim
        self.trajectory_features = nn.Sequential(
            nn.Linear(200, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
        )
        self.projector = nn.Linear(512, common_dim)

    def set_common_dim(self, common_dim):
        self.projector = nn.Linear(512,  common_dim)
        self.common_dim = common_dim

    def forward(self, x):
        h = self.trajectory_features(x)
        return self.projector(h)
    

class MHDSoundProcessor(nn.Module):
    def __init__(self, common_dim):
        super(MHDSoundProcessor, self).__init__()
        self.common_dim = common_dim
        # Properties
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

        # Output layer of the network
        self.projector = nn.Linear(2048, common_dim)

    def set_common_dim(self, common_dim):
        self.projector = nn.Linear(2048,  common_dim)
        self.common_dim = common_dim

    def forward(self, x):
        h = self.sound_features(x)
        h = h.view(h.size(0), -1)
        return self.projector(h)


class MHDJointProcessor(nn.Module):
    def __init__(self, common_dim):
        super(MHDJointProcessor, self).__init__()
        self.common_dim = common_dim
        self.img_features = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.GELU(),
        )

        self.trajectory_features = nn.Sequential(
            nn.Linear(200, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
        )
        self.projector = nn.Linear(128 * 7 * 7 + 512, common_dim)

    def set_common_dim(self, common_dim):
        self.projector = nn.Linear(128 * 7 * 7 + 512,  common_dim)
        self.common_dim = common_dim

    def forward(self, x):
        x_img, x_trajectory = x['image'], x['trajectory']

        # Image
        h_img = self.img_features(x_img)
        h_img = h_img.view(h_img.size(0), -1)

        # Trajectory
        h_trajectory = self.trajectory_features(x_trajectory)

        return self.projector(torch.cat((h_img, h_trajectory), dim=-1))