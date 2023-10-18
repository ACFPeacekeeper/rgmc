import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import reduce

filter_base = 32

class MSCommonEncoder(nn.Module):
    def __init__(self, common_dim, latent_dimension):
        super(MSCommonEncoder, self).__init__()
        self.common_dim = common_dim
        self.latent_dim = latent_dimension
        self.feature_extractor = nn.Sequential(nn.Linear(common_dim, 128), nn.GELU(), nn.Linear(128, latent_dimension),)

    def set_latent_dim(self, latent_dim):
        self.feature_extractor = nn.Sequential(nn.Linear(self.common_dim, 128), nn.GELU(), nn.Linear(128, latent_dim),)
        self.latent_dim = latent_dim

    def set_common_dim(self, common_dim):
        self.feature_extractor = nn.Sequential(nn.Linear(common_dim, 128), nn.GELU(), nn.Linear(128, self.latent_dim),)
        self.common_dim = common_dim

    def forward(self, x):
        return F.normalize(self.feature_extractor(x), dim=-1)


class MSCommonDecoder(nn.Module):
    def __init__(self, common_dim, latent_dimension):
        super(MSCommonDecoder, self).__init__()
        self.common_dim = common_dim
        self.latent_dim = latent_dimension
        self.feature_reconstructor = nn.Sequential(nn.Linear(latent_dimension, 128), nn.GELU(), nn.Linear(128, common_dim),)

    def set_latent_dim(self, latent_dim):
        self.feature_extractor = nn.Sequential(nn.Linear(latent_dim, 128), nn.GELU(), nn.Linear(128, self.common_dim),)
        self.latent_dim = latent_dim

    def set_common_dim(self, common_dim):
        self.feature_extractor = nn.Sequential(nn.Linear(self.latent_dim, 128), nn.GELU(), nn.Linear(128, common_dim),)
        self.common_dim = common_dim

    def forward(self, z):
        return self.feature_reconstructor(z)
    

class MSMNISTProcessor(nn.Module):
    def __init__(self, common_dim, dim):
        super(MSMNISTProcessor, self).__init__()
        self.dim = dim
        self.common_dim = common_dim
        self.mnist_features = nn.Sequential(
            nn.Conv2d(1, filter_base * 2, 4, 2, 1),
            nn.GELU(),
            nn.Conv2d(filter_base * 2, filter_base * 4, 4, 2, 1),
            nn.GELU(),
        )
        self.projector = nn.Linear(self.dim, common_dim)

    def set_common_dim(self, common_dim):
        self.projector = nn.Linear(self.dim,  common_dim)
        self.common_dim = common_dim

    def forward(self, x):
        h = self.mnist_features(x)
        h = h.view(h.size(0), -1)
        return self.projector(h)


class MSMNISTDecoder(nn.Module):
    def __init__(self, common_dim, dims):
        super(MSMNISTDecoder, self).__init__()
        self.dims = dims
        self.common_dim = common_dim
        self.projector = nn.Linear(common_dim, reduce(lambda x, y: x * y, self.dims))
        self.mnist_reconstructor = nn.Sequential(
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.GELU(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1)
        )

    def set_common_dim(self, common_dim):
        self.projector = nn.Linear(common_dim, reduce(lambda x, y: x * y, self.dims))
        self.common_dim = common_dim

    def forward(self, z):
        x_hat = self.projector(z)
        x_hat = x_hat.view(x_hat.size(0), *self.dims)
        return self.mnist_reconstructor(x_hat)


class MSSVHNProcessor(nn.Module):
    def __init__(self, common_dim, dim):
        super(MSSVHNProcessor, self).__init__()
        self.dim = dim
        self.common_dim = common_dim
        self.svhn_features = nn.Sequential(
            nn.Conv2d(3, filter_base * 2, 4, 2, 1),
            nn.GELU(),
            nn.Conv2d(filter_base * 2, filter_base * 4, 4, 2, 1),
            nn.GELU(),
        )
        self.projector = nn.Linear(self.dim, common_dim)

    def set_common_dim(self, common_dim):
        self.projector = nn.Linear(self.dim, common_dim)
        self.common_dim = common_dim

    def forward(self, x):
        h = self.svhn_features(x)
        h = h.view(h.size(0), -1)
        return self.projector(h)

class MSSVHNDecoder(nn.Module):
    def __init__(self, common_dim, dim):
        super(MSSVHNDecoder, self).__init__()
        self.dim = dim
        self.common_dim = common_dim
        self.projector = nn.Linear(common_dim, self.dim)
        self.svhn_reconstructor = nn.Sequential(
            nn.GELU(),
            nn.ConvTranspose2d(filter_base * 4, filter_base * 2, 4, 2, 1),
            nn.GELU(),
            nn.ConvTranspose2d(filter_base * 2, 3, 4, 2, 1),
        )

    def set_common_dim(self, common_dim):
        self.projector = nn.Linear(common_dim, self.dim)
        self.common_dim = common_dim

    def forward(self, h):
        x_hat = self.projector(h)
        return self.svhn_reconstructor(x_hat)        


class MSJointProcessor(nn.Module):
    def __init__(self, common_dim, mnist_dims, svhn_dims):
        super(MSJointProcessor, self).__init__()
        self.svhn_dims = svhn_dims
        self.mnist_dims = mnist_dims
        self.common_dim = common_dim
        self.mnist_features = nn.Sequential(
            nn.Conv2d(1, filter_base * 2, 4, 2, 1),
            nn.GELU(),
            nn.Conv2d(filter_base * 2, filter_base * 4, 4, 2, 1),
            nn.GELU(),
        )

        self.svhn_features = nn.Sequential(
            nn.Conv2d(3, filter_base * 2, 4, 2, 1),
            nn.GELU(),
            nn.Conv2d(filter_base * 2, filter_base * 4, 4, 2, 1),
            nn.GELU(),
        )

        self.projector = nn.Linear(reduce(lambda x, y: x * y, self.svhn_dims) + reduce(lambda x, y: x * y, self.mnist_dims), common_dim)

    def set_common_dim(self, common_dim):
        self.projector = nn.Linear(reduce(lambda x, y: x * y, self.svhn_dims) + reduce(lambda x, y: x * y, self.mnist_dims), common_dim)
        self.common_dim = common_dim

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
    def __init__(self, common_dim, mnist_dims, svhn_dims):
        super(MSJointDecoder, self).__init__()
        self.svhn_dims = svhn_dims
        self.mnist_dims = mnist_dims
        self.common_dim = common_dim
        self.projector = nn.Linear(common_dim, reduce(lambda x, y: x * y, self.svhn_dims) + reduce(lambda x, y: x * y, self.mnist_dims))
        self.mnist_reconstructor = nn.Sequential(
           nn.GELU(),
           nn.ConvTranspose2d(filter_base * 4, filter_base * 2, 4, 2, 1),
           nn.GELU(),
           nn.ConvTranspose2d(filter_base * 2, 1, 4, 2, 1), 
        )

        self.svhn_reconstructor = nn.Sequential(
            nn.GELU(),
            nn.ConvTranspose2d(filter_base * 4, filter_base * 2, 4, 2, 1),
            nn.GELU(),
            nn.ConvTranspose2d(filter_base * 2, 3, 4, 2, 1),
        )

    def set_common_dim(self, common_dim):
        self.projector = nn.Linear(common_dim, reduce(lambda x, y: x * y, self.svhn_dims) + reduce(lambda x, y: x * y, self.mnist_dims))
        self.common_dim = common_dim

    def forward(self, z):
        x_hat = self.projector(z)

        # MNIST recon
        mnist_dim = reduce(lambda x, y: x * y, self.mnist_dims)
        mnist_hat = x_hat[:, :mnist_dim]
        mnist_hat = mnist_hat.view(mnist_hat.size(0), *self.mnist_dims)
        mnist_hat = self.mnist_reconstructor(mnist_hat)

        # SVHN recon
        svhn_dim = reduce(lambda x, y: x * y, self.svhn_dims)
        svhn_hat = x_hat[:, mnist_dim:mnist_dim + svhn_dim]
        svhn_hat = svhn_hat.view(svhn_hat.size(0), *self.svhn_dims)
        svhn_hat = self.svhn_reconstructor(svhn_hat)

        return {"mnist": mnist_hat, "svhn": svhn_hat}