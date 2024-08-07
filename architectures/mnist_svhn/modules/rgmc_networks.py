import torch
import torch.nn as nn


filter_base = 32


class MSCommonEncoder(nn.Module):
    def __init__(self, common_dim, latent_dimension):
        super(MSCommonEncoder, self).__init__()
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


class MSMNISTProcessor(nn.Module):
    def __init__(self, common_dim, dim):
        super(MSMNISTProcessor, self).__init__()
        self.dim = dim
        self.common_dim = common_dim
        self.mnist_features = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
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


class MSSVHNProcessor(nn.Module):
    def __init__(self, common_dim, dim):
        super(MSSVHNProcessor, self).__init__()
        self.dim = dim
        self.common_dim = common_dim
        self.svhn_features = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(32, 32 * 2, 4, 2, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(32 * 2, 32 * 4, 4, 2, 1, bias=False),
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


class MSJointProcessor(nn.Module):
    def __init__(self, common_dim, mnist_dim, svhn_dim):
        super(MSJointProcessor, self).__init__()
        self.svhn_dim = svhn_dim
        self.mnist_dim = mnist_dim
        self.common_dim = common_dim
        self.mnist_features = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.GELU(),
        )

        self.svhn_features = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(32, 32 * 2, 4, 2, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(32 * 2, 32 * 4, 4, 2, 1, bias=False),
            nn.GELU(),
        )

        self.projector = nn.Linear(self.mnist_dim + self.svhn_dim, common_dim)

    def set_common_dim(self, common_dim):
        self.projector = nn.Linear(self.mnist_dim + self.svhn_dim, common_dim)
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
    

class OddOneOutNetwork(nn.Module):
    def __init__(self, latent_dim, num_modalities, modalities, device):
        super().__init__()
        self.latent_dimension = latent_dim
        self.num_modalities = num_modalities
        self.modalities = modalities
        self.device = device
        self.embedder = nn.Sequential(
            nn.Conv1d(num_modalities, 64, 4, 2, 1),
            nn.GELU(),
            nn.Conv1d(64, 128, 4, 2, 1),
            nn.GELU(),
            nn.Dropout(),
        )
        self.clf_fc = nn.Linear(2048, num_modalities + 1)
        self.classificator = nn.Softmax(dim=-1)

    def set_latent_dim(self, latent_dim):
        self.latent_dimension = latent_dim

    def forward(self, mod_representations):
        batch_size = mod_representations[0].size()[0]
        representations = mod_representations[0].view(batch_size, 1, self.latent_dimension)
        for mod_rep in mod_representations[1:]:
            mod_rep = mod_rep.view(batch_size, 1, self.latent_dimension)
            representations = torch.cat((representations, mod_rep), dim=1)
        h = self.embedder(representations)
        h = self.clf_fc(h.view(h.size(0), -1))
        classes = self.classificator(h)
        return classes