import torch

from torch.nn import ReLU
from collections import Counter
from ..subnetworks.vae_networks import *


class MSVAE(nn.Module):
    def __init__(self, name, latent_dimension, device, exclude_modality, scales, mean, std):
        super(MSVAE, self).__init__()
        self.name = name
        self.layer_dim = 28 * 28 + 3 * 32 * 32
        self.modality_dims = [0, 28 * 28, 3 * 32 * 32]
        self.exclude_modality = exclude_modality
        self.latent_dimension = latent_dimension
        self.encoder = Encoder(self.latent_dimension, self.layer_dim)
        self.decoder = Decoder(self.latent_dimension, self.layer_dim)
        self.inf_activation = ReLU()
        self.device = device
        self.scales = scales
        self.mean = mean
        self.std = std
        self.kld = 0.

    def set_modalities(self, exclude_modality):
        self.exclude_modality = exclude_modality

    def set_latent_dim(self, latent_dim):
        self.encoder.set_latent_dim(latent_dim)
        self.decoder.set_latent_dim(latent_dim)
        self.latent_dimension = latent_dim

    def reparameterization(self, mean, std):
        dist = torch.distributions.Normal(self.mean, self.std)
        eps = dist.sample(std.shape).to(self.device)
        z = torch.add(mean, torch.mul(std, eps))
        return z

    def forward(self, x, sample=False):
        data_list = list(x.values())
        data = torch.flatten(data_list[0], start_dim=1)

        for id in range(1, len(data_list)):
            data_list[id] = torch.flatten(data_list[id], start_dim=1)
            data = torch.concat((data, data_list[id]), dim=-1)

        mean, logvar = self.encoder(data)
        std = torch.exp(torch.mul(logvar, 0.5))
        if sample is False and not isinstance(self.scales['kld_beta'], type(None)):
            z = self.reparameterization(mean, std)
            self.kld = - self.scales['kld_beta'] * torch.mean(1 + logvar - mean.pow(2) - std.pow(2)) * (self.latent_dimension / data_list[0].size(dim=0))
        else:
            z = mean
        
        tmp = self.decoder(z)
        x_hat = dict.fromkeys(x.keys())
        for id, key in enumerate(x_hat.keys()):
            x_hat[key] = tmp[:, self.modality_dims[id]:self.modality_dims[id]+self.modality_dims[id+1]]
            if key == 'mnist':
                x_hat[key] = torch.reshape(x_hat[key], (x_hat[key].size(dim=0), 1, 28, 28))
            elif key == 'svhn':
                x_hat[key] = torch.reshape(x_hat[key], (x_hat[key].size(dim=0), 3, 32, 32))

        return x_hat, z
    
    
    def loss(self, x, x_hat):
        mse_loss = nn.MSELoss(reduction="none").to(self.device)
        recon_losses =  dict.fromkeys(x.keys())

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
        x_hat, _ = self.forward(x, sample=True)
        elbo, loss_dict = self.loss(x, x_hat)
        return elbo, loss_dict
    
    def inference(self, x, labels):
        x_hat, z = self.forward(x, sample=True)
        for key in x_hat.keys():
            x_hat[key] = self.inf_activation(x_hat)
        
        return z, x_hat