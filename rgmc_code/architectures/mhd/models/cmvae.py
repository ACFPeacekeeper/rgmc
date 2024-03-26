import torch

from torch.nn import ReLU
from collections import Counter
from ..subnetworks.cmvae_networks import *


class MHDCMVAE(nn.Module):
    def __init__(self, name, latent_dimension, device, exclude_modality, scales, mean, std):
        super(MHDCMVAE, self).__init__()
        self.kld = 0.
        self.std = std
        self.mean = mean
        self.name = name
        self.device = device
        self.scales = scales
        self.exclude_modality = exclude_modality
        self.latent_dimension = latent_dimension
        self.inf_activation = ReLU()
        self.trajectory_encoder = TrajectoryEncoder(latent_dimension)
        self.trajectory_decoder = TrajectoryDecoder(latent_dimension)
        self.image_encoder = ImageEncoder(latent_dimension)
        self.image_decoder = ImageDecoder(latent_dimension)
        self.common_encoder = CommonEncoder(latent_dimension)
        self.common_decoder = CommonDecoder(latent_dimension)
        self.encoders = {'image': self.image_encoder, 'trajectory': self.trajectory_encoder}
        self.decoders = {'image': self.image_decoder, 'trajectory': self.trajectory_decoder}

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

    def forward(self, x, sample=False):
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

        loss_dict = Counter({'total_loss': recon_loss, 'img_recon_loss': recon_losses['image'], 'traj_recon_loss': recon_losses['trajectory']})
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