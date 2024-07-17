import torch
import collections
import torch.nn as nn

from ..modules.cmdvae_networks import (
    MNISTEncoder, MNISTDecoder,
    SVHNEncoder, SVHNDecoder,
    CommonEncoder, CommonDecoder
)


class MSCMDVAE(nn.Module):
    def __init__(self, name, latent_dimension, device, exclude_modality, scales, mean, std, noise_factor):
        super(MSCMDVAE, self).__init__()
        self.kld = 0.
        self.std = std
        self.mean = mean
        self.name = name
        self.device = device
        self.scales = scales
        self.noise_factor = noise_factor
        self.exclude_modality = exclude_modality
        self.latent_dimension = latent_dimension
        self.inf_activation = nn.ReLU()
        self.mnist_encoder = MNISTEncoder(latent_dimension)
        self.mnist_decoder = MNISTDecoder(latent_dimension)
        self.svhn_encoder = SVHNEncoder(latent_dimension)
        self.svhn_decoder = SVHNDecoder(latent_dimension)
        self.common_encoder = CommonEncoder(latent_dimension)
        self.common_decoder = CommonDecoder(latent_dimension)
        self.encoders = {'mnist': self.mnist_encoder, 'svhn': self.svhn_encoder}
        self.decoders = {'mnist': self.mnist_decoder, 'svhn': self.svhn_decoder}

    def set_latent_dim(self, latent_dim):
        self.common_encoder.set_latent_dim(latent_dim)
        self.common_decoder.set_latent_dim(latent_dim)
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
    
    def add_noise(self, x):
        x_noisy = dict.fromkeys(x.keys())
        for key, modality in x.items():
            x_noisy[key] = torch.clamp(torch.add(modality, torch.mul(torch.randn_like(modality), self.noise_factor)), torch.min(modality), torch.max(modality))
        return x_noisy

    def forward(self, x, sample=False):
        if sample is False and self.noise_factor != 0:
            x = self.add_noise(x)
            
        batch_size = list(x.values())[0].size(dim=0)
        latent_reps = []
        for key in x.keys():
            if key == self.exclude_modality:
                continue
            latent_reps.append(self.encoders[key](x[key]))

        mean, logvar = self.common_encoder(torch.stack(latent_reps, dim=0).sum(dim=0))
        std = torch.exp(torch.mul(logvar, 0.5))
        if sample is False and not isinstance(self.scales['kld_beta'], type(None)):
            z = self.reparameterization(mean, std)
            self.kld = - self.scales['kld_beta'] * torch.mean(1 + logvar - mean.pow(2) - std.pow(2)) * (self.latent_dimension / batch_size)
        else:
            z = mean
        
        tmp = self.common_decoder(z)
        x_hat = dict.fromkeys(x.keys())
        for key in x_hat.keys():
            x_hat[key] = self.decoders[key](tmp)

        return x_hat, z
    
    def loss(self, x, x_hat):
        mse_loss = nn.MSELoss(reduction="none").to(self.device)
        recon_losses = dict.fromkeys(x.keys())

        for key in x.keys():
            loss = mse_loss(x_hat[key], x[key])
            recon_losses[key] = self.scales[key] * (loss / torch.as_tensor(loss.size()).prod().sqrt()).sum() 

        recon_loss = 0
        for value in recon_losses.values():
            recon_loss += value

        loss_dict = collections.Counter({'total_loss': recon_loss, 'mnist_recon_loss': recon_losses['mnist'], 'svhn_recon_loss': recon_losses['svhn']})
        return recon_loss, loss_dict

    def training_step(self, x, labels):
        x_hat, _ = self.forward(x, sample=False)
        elbo, loss_dict = self.loss(x, x_hat)
        return elbo, loss_dict
    
    def validation_step(self, x, labels):
        x_hat, _ = self.forward(x, sample=True)
        elbo, loss_dict = self.loss(x, x_hat)
        return elbo, loss_dict
    
    def inference(self, x, labels):
        x_hat, z = self.forward(x, sample=True)
        for key in x_hat.keys():
            x_hat[key] = torch.clamp(x_hat[key], torch.min(x[key]), torch.max(x[key]))
        
        return z, x_hat