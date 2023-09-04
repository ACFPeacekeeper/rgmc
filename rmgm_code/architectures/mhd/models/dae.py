import torch

from collections import Counter
from ..subnetworks.dae_networks import *


class MhdDAE(nn.Module):
    def __init__(self, name, latent_dimension, device, exclude_modality, scales, noise_factor=0.3):
        super(MhdDAE, self).__init__()
        self.name = name
        self.layer_dim = 28 * 28 + 200
        self.modality_dims = [0, 28 * 28, 200]
        self.exclude_modality = exclude_modality

        self.exclude_modality
        self.encoder = Encoder(latent_dimension, self.layer_dim)
        self.decoder = Decoder(latent_dimension, self.layer_dim)
        self.latent_dimension = latent_dimension
        self.device = device
        self.scales = scales
        self.noise_factor = noise_factor

    def set_modalities(self, exclude_modality):
        self.exclude_modality = exclude_modality

    def set_latent_dim(self, latent_dim):
        self.encoder.set_latent_dim(latent_dim)
        self.decoder.set_latent_dim(latent_dim)
        self.latent_dimension = latent_dim

    def add_noise(self, x):
        x_noisy = dict.fromkeys(x.keys())
        for key, modality in x.items():
            x_noisy[key] = torch.clamp(torch.add(modality, torch.mul(torch.randn_like(modality), self.noise_factor)), torch.min(modality), torch.max(modality))
        return x_noisy

    def forward(self, x, sample=False):
        if sample is False and self.noise_factor != 0:
            x = self.add_noise(x)

        data_list = list(x.values())
        if len(data_list[0].size()) > 2:
            data = torch.flatten(data_list[0], start_dim=1)
        else:
            data = data_list[0]

        for id in range(1, len(data_list)):
            if len(data_list[id].size()) > 2:
                data_list[id] = torch.flatten(data_list[id], start_dim=1)
            data = torch.concat((data, data_list[id]), dim=-1)

        z = self.encoder(data)
        tmp = self.decoder(z)

        x_hat = dict.fromkeys(x.keys())
        for id, key in enumerate(x_hat.keys()):
            x_hat[key] = tmp[:, self.modality_dims[id]:self.modality_dims[id]+self.modality_dims[id+1]]
            if key == 'image':
                x_hat[key] = torch.reshape(x_hat[key], (x_hat[key].size(dim=0), 1, 28, 28))

        return x_hat, z
    
    def loss(self, x, x_hat):
        mse_loss = nn.MSELoss(reduction="none").to(self.device)
        recon_losses =  dict.fromkeys(x.keys())

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
            recon_loss, loss_dict = self.loss(x, x_hat)
            return recon_loss, loss_dict
    
    def validation_step(self, x, labels):
        return self.training_step(x, labels)
    