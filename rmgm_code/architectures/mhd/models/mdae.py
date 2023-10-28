import torch

from torch.nn import ReLU
from collections import Counter
from ..subnetworks.mdae_networks import *


class MhdMDAE(nn.Module):
    def __init__(self, name, latent_dimension, device, exclude_modality, scales, noise_factor=0.5):
        super(MhdMDAE, self).__init__()
        self.name = name
        self.device = device
        self.noise_factor = noise_factor
        self.scales = scales
        self.exclude_modality = exclude_modality
        self.latent_dimension = latent_dimension
        self.inf_activation = ReLU()
        self.image_encoder = None
        self.image_decoder = None
        self.trajectory_encoder = None
        self.trajectory_decoder = None
        self.trajectory_encoder = TrajectoryEncoder(latent_dimension)
        self.trajectory_decoder = TrajectoryDecoder(latent_dimension)
        self.image_encoder = ImageEncoder(latent_dimension)
        self.image_decoder = ImageDecoder(latent_dimension)
        self.encoders = {'image': self.image_encoder, 'trajectory': self.trajectory_encoder}
        self.decoders = {'image': self.image_decoder, 'trajectory': self.trajectory_decoder}

    def set_latent_dim(self, latent_dim):
        for enc_key, dec_key in zip(self.encoders.keys(), self.decoders.keys()):
            self.encoders[enc_key].set_latent_dim(latent_dim)
            self.decoders[dec_key].set_latent_dim(latent_dim)
        self.latent_dimension = latent_dim

    def set_modalities(self, exclude_modality):
        self.exclude_modality = exclude_modality

    def add_noise(self, x):
        x_noisy = dict.fromkeys(x.keys())
        for key, modality in x.items():
            x_noisy[key] = torch.clamp(torch.add(modality, torch.mul(torch.randn_like(modality), self.noise_factor)), torch.min(modality), torch.max(modality))
        return x_noisy

    def forward(self, x, sample=False):
        if sample is False and self.noise_factor != 0:
            x = self.add_noise(x)

        latent_reps = []
        for key in x.keys():
            if key == self.exclude_modality:
                continue
            latent_reps.append(self.encoders[key](x[key]))

        z = torch.stack(latent_reps, dim=0).sum(dim=0) / len(latent_reps)

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