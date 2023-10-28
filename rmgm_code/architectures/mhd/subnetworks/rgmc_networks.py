import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import reduce

filter_base = 32

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
        return F.normalize(self.latent_fc(self.feature_extractor(h)), dim=-1)


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
    

class MHDImageProcessor(nn.Module):
    def __init__(self, common_dim, dim):
        super(MHDImageProcessor, self).__init__()
        self.dim = dim
        self.common_dim = common_dim
        self.image_features = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.GELU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.GELU(),
        )
        self.projector = nn.Linear(dim, common_dim)

    def set_common_dim(self, common_dim):
        self.projector = nn.Linear(self.dim,  common_dim)
        self.common_dim = common_dim

    def forward(self, x):
        h = self.image_features(x)
        h = h.view(h.size(0), -1)
        return self.projector(h)


class MHDTrajectoryProcessor(nn.Module):
    def __init__(self, common_dim, dim):
        super(MHDTrajectoryProcessor, self).__init__()
        self.dim = dim
        self.trajectory_features = nn.Sequential(
            nn.Linear(200, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
        )
        self.projector = nn.Linear(self.dim, common_dim)

    def set_common_dim(self, common_dim):
        self.projector = nn.Linear(self.dim,  common_dim)
        self.common_dim = common_dim

    def forward(self, x):
        h = self.trajectory_features(x)
        return self.projector(h)


class MHDJointProcessor(nn.Module):
    def __init__(self, common_dim, image_dim, traj_dim):
        super(MHDJointProcessor, self).__init__()
        self.traj_dim = traj_dim
        self.image_dim = image_dim
        self.common_dim = common_dim
        self.img_features = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.GELU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.GELU(),
        )

        self.trajectory_features = nn.Sequential(
            nn.Linear(200, 512),
            nn.GELU(),
            nn.Linear(512, self.traj_dim),
            nn.GELU(),
        )
        self.projector = nn.Linear(self.image_dim + self.traj_dim, common_dim)

    def set_common_dim(self, common_dim):
        self.projector = nn.Linear(self.image_dim + self.traj_dim, common_dim)
        self.common_dim = common_dim

    def forward(self, x):
        x_img, x_trajectory = x['image'], x['trajectory']

        # Image
        h_img = self.img_features(x_img)
        h_img = h_img.view(h_img.size(0), -1)

        # Trajectory
        h_trajectory = self.trajectory_features(x_trajectory)

        return self.projector(torch.cat((h_img, h_trajectory), dim=-1))