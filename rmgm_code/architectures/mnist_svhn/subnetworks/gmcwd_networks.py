import torch
import torch.nn as nn
import torch.nn.functional as F


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


class MSCommonDecoder(nn.Module):
    def __init__(self, common_dim, latent_dimension):
        super(MSCommonDecoder, self).__init__()
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

    def forward(self, z):
        h = self.latent_fc(z)
        return self.feature_reconstructor(h)
    

class MSMNISTProcessor(nn.Module):
    def __init__(self, common_dim):
        super(MSMNISTProcessor, self).__init__()
        self.common_dim = common_dim
        self.mnist_features = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            Swish(),
            nn.Conv2d(64, 128, 4, 2, 1),
            Swish(),
        )
        self.projector = nn.Linear(128 * 7 * 7, common_dim)

    def forward(self, x):
        h = self.mnist_features(x)
        h = h.view(h.size(0), -1)
        return self.projector(h)


class MSMNISTDecoder(nn.Module):
    def __init__(self, common_dim):
        super(MSMNISTDecoder, self).__init__()
        self.common_dim = common_dim
        self.projector = nn.Linear(common_dim, 128 * 7 * 7)
        self.mnist_reconstructor = nn.Sequential(
            Swish(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            Swish(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1)
        )

    def forward(self, z):
        x_hat = self.projector(z)
        x_hat = x_hat.view(x_hat.size(0), 128, 7, 7)
        return self.mnist_reconstructor(x_hat)


class MSSVHNProcessor(nn.Module):
    def __init__(self, common_dim):
        super(MSSVHNProcessor, self).__init__()
        self.common_dim = common_dim
        self.svhn_features = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            Swish(),
            nn.Conv2d(32, 32 * 2, 4, 2, 1),
            Swish(),
            nn.Conv2d(32 * 2, 32 * 4, 4, 2, 1),
            Swish(),
        )
        self.projector = nn.Linear(32 * 32 * 2, common_dim)

    def forward(self, x):
        h = self.svhn_features(x)
        h = h.view(h.size(0), -1)
        return self.projector(h)

class MSSVHNDecoder(nn.Module):
    def __init__(self, common_dim):
        super(MSSVHNDecoder, self).__init__()
        self.common_dim = common_dim
        self.projector = nn.Linear(common_dim, 32 * 32 * 2)
        self.svhn_reconstructor = nn.Sequential(
            nn.ConvTranspose2d(32 * 4, 32 * 2, 4, 2, 1),
            Swish(),
            nn.ConvTranspose2d(32 * 2, 32, 4, 2, 1),
            Swish(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
        )

    def forward(self, h):
        x_hat = self.projector(h)
        return self.svhn_reconstructor(x_hat)        


class MSLabelProcessor(nn.Module):
    def __init__(self, common_dim):
        super(MSLabelProcessor, self).__init__()
        self.common_dim = common_dim
        self.projector = nn.Linear(10, common_dim)

    def forward(self, x):
        return self.projector(x)


class MSLabelDecoder(nn.Module):
    def __init__(self, common_dim):
        super(MSLabelDecoder, self).__init__()
        self.common_dim = common_dim
        self.projector = nn.Linear(common_dim, 10)

    def forward(self, h):
        return self.projector(h)


class MSJointProcessor(nn.Module):
    def __init__(self, common_dim):
        super(MSJointProcessor, self).__init__()
        self.common_dim = common_dim

        self.mnist_features = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            Swish(),
            nn.Conv2d(64, 128, 4, 2, 1),
            Swish(),
        )

        self.svhn_features = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            Swish(),
            nn.Conv2d(32, 32 * 2, 4, 2, 1),
            Swish(),
            nn.Conv2d(32 * 2, 32 * 4, 4, 2, 1),
            Swish(),
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


class MSJointDecoder(nn.Module):
    def __init__(self, common_dim):
        super(MSJointDecoder, self).__init__()
        self.common_dim = common_dim
        self.projector = nn.Linear(common_dim, 128 * 7 * 7 + 32 * 32 * 2)
        self.mnist_reconstructor = nn.Sequential(
           Swish(),
           nn.ConvTranspose2d(128, 64, 4, 2, 1),
           Swish(),
           nn.ConvTranspose2d(64, 1, 4, 2, 1), 
        )

        self.svhn_reconstructor = nn.Sequential(
            nn.ConvTranspose2d(32 * 4, 32 * 2, 4, 2, 1),
            Swish(),
            nn.ConvTranspose2d(32 * 2, 32, 4, 2, 1),
            Swish(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
        )

    def forward(self, z):
        x_hat = self.projector(z)

        # MNIST recon
        mnist_hat = x_hat[:, :128 * 7 * 7]
        mnist_hat = mnist_hat.view(mnist_hat.size(0), 128, 7, 7)
        mnist_hat = self.mnist_reconstructor(mnist_hat)

        # SVHN recon
        svhn_hat = x_hat[:, 128 * 7 * 7:128 * 7 * 7 + 32 * 32 * 2]
        print(svhn_hat.size())
        svhn_hat = svhn_hat.view(svhn_hat.size(0), 128, 4, 4)
        svhn_hat = self.svhn_reconstructor(svhn_hat)

        return {"mnist": mnist_hat, "svhn": svhn_hat}

"""


Extra components


"""

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
