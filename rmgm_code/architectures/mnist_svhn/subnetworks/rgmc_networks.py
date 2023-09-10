import torch
import torch.nn as nn
import torch.nn.functional as F


# Code adapted from https://github.com/miguelsvasco/gmc
class MSCommonEncoder(nn.Module):
    def __init__(self, common_dim, latent_dimension):
        super(MSCommonEncoder, self).__init__()
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


class MSMNISTProcessor(nn.Module):
    def __init__(self, common_dim):
        super(MSMNISTProcessor, self).__init__()
        self.common_dim = common_dim
        self.mnist_features = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            Swish(),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            Swish(),
        )
        self.projector = nn.Linear(128 * 7 * 7, common_dim)

    def forward(self, x):
        h = self.mnist_features(x)
        h = h.view(h.size(0), -1)
        return self.projector(h)


class MSSVHNProcessor(nn.Module):
    def __init__(self, common_dim):
        super(MSSVHNProcessor, self).__init__()
        self.common_dim = common_dim
        self.svhn_features = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(32, 32 * 2, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(32 * 2, 32 * 4, 4, 2, 1),
            nn.SiLU(),
        )
        self.projector = nn.Linear(32 * 32 * 2, common_dim)

    def forward(self, x):
        h = self.svhn_features(x)
        h = h.view(h.size(0), -1)
        return self.projector(h)


class MSLabelProcessor(nn.Module):
    def __init__(self, common_dim):
        super(MSLabelProcessor, self).__init__()
        self.common_dim = common_dim
        self.projector = nn.Linear(10, common_dim)

    def forward(self, x):
        return self.projector(x)



class MSJointProcessor(nn.Module):
    def __init__(self, common_dim):
        super(MSJointProcessor, self).__init__()
        self.common_dim = common_dim

        self.mnist_features = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            Swish(),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            Swish(),
        )

        self.svhn_features = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(32, 32 * 2, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(32 * 2, 32 * 4, 4, 2, 1),
            nn.SiLU(),
        )

        self.projector = nn.Linear(128 * 7 * 7 + 32 * 32 * 2, common_dim)

    def forward(self, x):
        x_mnist, x_svhn = x['mnist'], x['svhn']

        # MNIST
        h_mnist = self.mnist_features(x_mnist)
        h_mnist = h_mnist.view(h_mnist.size(0), -1)

        # SVHN
        h_svhn = self.svhn_features(x_svhn)
        h_svhn = h_svhn.view(h_svhn.size(0), -1)

        return self.projector(torch.cat((h_mnist, h_svhn), dim=-1))



"""


Extra components


"""

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
