import torch
import torch.nn as nn

from collections import Counter
from torch.autograd import Variable
from architectures.mvae_networks import ImageEncoder, ImageDecoder, TrajectoryEncoder, TrajectoryDecoder

class MVAE(nn.Module):
    def __init__(self, name, latent_dim, device, exclude_modality, scales, mean, std, expert_type, test=False):
        super(MVAE, self).__init__()
        self.name = name
        self.device = device
        self.mean = mean
        self.std = std
        self.scales = scales
        self.exclude_modality = exclude_modality
        self.latent_dim = latent_dim
        self.kld = 0.
        self.experts = PoE() if expert_type == 'PoE' else PoE()
        self.image_encoder = None
        self.image_decoder = None
        self.trajectory_encoder = None
        self.trajectory_decoder = None

        if self.exclude_modality == 'image':
            self.trajectory_encoder = TrajectoryEncoder(latent_dim)
            self.trajectory_decoder = TrajectoryDecoder(latent_dim)
            self.encoders = {'trajectory': self.trajectory_encoder}
            self.decoders = {'trajectory': self.trajectory_decoder}
        elif self.exclude_modality == 'trajectory':
            self.image_encoder = ImageEncoder(latent_dim)
            self.image_decoder = ImageDecoder(latent_dim)
            self.encoders = {'image': self.image_encoder}
            self.decoders = {'image': self.image_decoder}
        else:
            self.trajectory_encoder = TrajectoryEncoder(latent_dim)
            self.trajectory_decoder = TrajectoryDecoder(latent_dim)
            self.image_encoder = ImageEncoder(latent_dim)
            self.image_decoder = ImageDecoder(latent_dim)
            self.encoders = {'image': self.image_encoder, 'trajectory': self.trajectory_encoder}
            self.decoders = {'image': self.image_decoder, 'trajectory': self.trajectory_decoder}


    def set_modalities(self, exclude_modality):
        if self.exclude_modality == 'image':
            self.trajectory_encoder = TrajectoryEncoder(self.latent_dim)
            self.trajectory_decoder = TrajectoryDecoder(self.latent_dim)
            self.encoders = {'trajectory': self.trajectory_encoder}
            self.decoders = {'trajectory': self.trajectory_decoder}
        elif self.exclude_modality == 'trajectory':
            self.image_encoder = ImageEncoder(self.latent_dim)
            self.image_decoder = ImageDecoder(self.latent_dim)
            self.encoders = {'image': self.image_encoder}
            self.decoders = {'image': self.image_decoder}
        else:
            self.trajectory_encoder = TrajectoryEncoder(self.latent_dim)
            self.trajectory_decoder = TrajectoryDecoder(self.latent_dim)
            self.image_encoder = ImageEncoder(self.latent_dim)
            self.image_decoder = ImageDecoder(self.latent_dim)
            self.encoders = {'image': self.image_encoder, 'trajectory': self.trajectory_encoder}
            self.decoders = {'image': self.image_decoder, 'trajectory': self.trajectory_decoder}

        self.exclude_modality = exclude_modality

    def forward(self, x):
        batch_size = list(x.values())[0].size(dim=0)
        mean, logvar = self.experts.prior_expert((1, batch_size, self.latent_dim), self.device)

        for key in x.keys():
            tmp_mean, tmp_logvar = self.encoders[key](x[key])
            mean = torch.cat((mean, tmp_mean.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, tmp_logvar.unsqueeze(0)), dim=0)

        mean, logvar = self.experts(mean, logvar)

        # Reparameterization trick
        std = torch.exp(torch.mul(logvar, 0.5))
        dist = torch.distributions.Normal(self.mean, self.std)
        eps = dist.sample(mean.shape).to(self.device)

        z = torch.add(mean, torch.mul(std, eps))

        self.kld = - self.scales['kld beta'] * (torch.sum(1 + logvar - mean.pow(2) - std.pow(2)) / batch_size)

        x_hat = dict.fromkeys(x.keys())
        for key in x_hat.keys():
            x_hat[key] = self.decoders[key](z)

        return x_hat, z
    
    def loss(self, x, x_hat):
        loss_function = nn.MSELoss().to(self.device)
        recon_losses =  dict.fromkeys(x.keys())

        for key in x.keys():
            recon_losses[key] = self.scales[key] * loss_function(x_hat[key], x[key])

        elbo = self.kld + torch.stack(list(recon_losses.values())).sum()

        if self.exclude_modality == 'trajectory':
            recon_losses['trajectory'] = 0.
        elif self.exclude_modality == 'image':
            recon_losses['image'] = 0.

        loss_dict = Counter({'ELBO loss': elbo, 'KLD loss': self.kld, 'Img recon loss': recon_losses['image'], 'Traj recon loss': recon_losses['trajectory']})
        self.kld = 0.
        return elbo, loss_dict
            

class PoE(nn.Module): 
    def forward(self, mean, logvar, eps=1e-8):
        var       = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T         = 1. / (var + eps)
        pd_mu     = torch.sum(mean * T, dim=0) / torch.sum(T, dim=0)
        pd_var    = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var + eps)
        return pd_mu, pd_logvar 

    def prior_expert(self, size, device):
        mean   = Variable(torch.zeros(size)).to(device)
        logvar = Variable(torch.zeros(size)).to(device)
        return mean, logvar
    
