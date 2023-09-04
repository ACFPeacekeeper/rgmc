import torch

from collections import Counter
from torch.autograd import Variable
from ..subnetworks.mvae_networks import *


class MSMVAE(nn.Module):
    def __init__(self, name, latent_dimension, device, exclude_modality, scales, mean, std, expert_type, poe_eps):
        super(MSMVAE, self).__init__()
        self.name = name
        self.device = device
        self.mean = mean
        self.std = std
        self.scales = scales
        self.exclude_modality = exclude_modality
        self.latent_dimension = latent_dimension
        self.kld = 0.
        self.experts = PoE() if expert_type == 'PoE' else PoE()
        self.poe_eps = poe_eps
        self.image_encoder = None
        self.image_decoder = None
        self.trajectory_encoder = None
        self.trajectory_decoder = None

        if self.exclude_modality == 'mnist':
            self.svhn_encoder = SVHNEncoder(latent_dimension)
            self.svhn_decoder = SVHNDecoder(latent_dimension)
            self.encoders = {'svhn': self.svhn_encoder}
            self.decoders = {'svhn': self.svhn_decoder}
        elif self.exclude_modality == 'svhn':
            self.mnist_encoder = MNISTEncoder(latent_dimension)
            self.mnist_decoder = MNISTDecoder(latent_dimension)
            self.encoders = {'mnist': self.mnist_encoder}
            self.decoders = {'mnist': self.mnist_decoder}
        else:
            self.svhn_encoder = SVHNEncoder(latent_dimension)
            self.svhn_decoder = SVHNDecoder(latent_dimension)
            self.mnist_encoder = MNISTEncoder(latent_dimension)
            self.mnist_decoder = MNISTDecoder(latent_dimension)
            self.encoders = {'mnist': self.mnist_encoder, 'svhn': self.svhn_encoder}
            self.decoders = {'mnist': self.mnist_decoder, 'svhn': self.svhn_decoder}

    def set_latent_dim(self, latent_dim):
        for enc_key, dec_key in zip(self.encoders.keys(), self.decoders.keys()):
            self.encoders[enc_key].set_latent_dim(latent_dim)
            self.decoders[dec_key].set_latent_dim(latent_dim)
        self.latent_dimension = latent_dim


    def set_modalities(self, exclude_modality):
        self.exclude_modality = exclude_modality

    def reparameterization(self, mean, std):
        dist = torch.distributions.Normal(self.mean, self.std)
        eps = dist.sample(std.shape).to(self.device)
        z = torch.add(mean, torch.mul(std, eps))
        return z

    def forward(self, x, sample=False):
        batch_size = list(x.values())[0].size(dim=0)
        mean, logvar = self.experts.prior_expert((1, batch_size, self.latent_dimension), self.device)

        for key in x.keys():
            if key == self.exclude_modality:
                continue
            tmp_mean, tmp_logvar = self.encoders[key](x[key])
            mean = torch.cat((mean, tmp_mean.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, tmp_logvar.unsqueeze(0)), dim=0)


        mean, logvar = self.experts(mean, logvar, self.poe_eps)
        std = torch.exp(torch.mul(logvar, 0.5))

        if sample is False:
            z = self.reparameterization(mean, std)
            self.kld = - self.scales['kld_beta'] * torch.mean(1 + logvar - mean.pow(2) - std.pow(2)) * (self.latent_dimension / batch_size)
        else:
            z = mean

        x_hat = dict.fromkeys(x.keys())
        for key in x_hat.keys():
            x_hat[key] = self.decoders[key](z)

        return x_hat, z
    
    def loss(self, x, x_hat):
        mse_loss = nn.MSELoss(reduction="none").to(self.device)
        recon_losses = dict.fromkeys(x.keys())

        for key in x.keys():
            loss = mse_loss(x_hat[key], x[key])
            recon_losses[key] = self.scales[key] * (loss / torch.as_tensor(loss.size()).prod().sqrt()).sum() 
        
        elbo = self.kld + torch.stack(list(recon_losses.values())).sum()

        loss_dict = Counter({'elbo_loss': elbo, 'kld_loss': self.kld, 'mnist_recon_loss': recon_losses['mnist'], 'svhn_recon_loss': recon_losses['svhn']})
        self.kld = 0.
        return elbo, loss_dict

    def training_step(self, x, labels):
        x_hat, _ = self.forward(x, sample=False)
        elbo, loss_dict = self.loss(x, x_hat)
        return elbo, loss_dict
    
    def validation_step(self, x, labels):
        return self.training_step(x, labels)
            

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
    
