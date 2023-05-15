import torch
import torch.nn as nn

from collections import Counter
from architectures.vae_networks import Encoder, Decoder

class VAE(nn.Module):
    def __init__(self, name, latent_dim, device, exclude_modality, scales, mean, std, test=False):
        super(VAE, self).__init__()
        self.name = name
        if exclude_modality == 'image':
            self.layer_dim = 200
            self.modality_dims = [0, 200]
        elif exclude_modality == 'trajectory':
            self.layer_dim = 28 * 28
            self.modality_dims = [0, 28 * 28]
        else:
            self.layer_dim = 28 * 28 + 200
            self.modality_dims = [0, 28 * 28, 200]

        self.encoder = Encoder(latent_dim, self.layer_dim)
        self.decoder = Decoder(latent_dim, self.layer_dim)
        
        self.device = device
        self.scales = scales
        self.mean = mean
        self.std = std
        self.kld = 0.
        if test:
            self.kld_scale = self.scales['kld beta'][1]
        else:
            self.kld_scale = self.scales['kld beta'][0]
        self.kld_max = self.scales['kld beta'][1]
        self.exclude_modality = exclude_modality

    def update_kld_scale(self, kld_weight):
        self.kld_scale = min(kld_weight * self.kld_max, self.kld_max)

    def set_modalities(self, exclude_modality):
        if exclude_modality == 'image':
            self.layer_dim = 200
            self.modality_dims = [0, 200]
        elif exclude_modality == 'trajectory':
            self.layer_dim = 28 * 28
            self.modality_dims = [0, 28 * 28]
        else:
            self.layer_dim = 28 * 28 + 200
            self.modality_dims = [0, 28 * 28, 200]

        self.exclude_modality = exclude_modality
        self.encoder.set_first_layer(self.layer_dim)
        self.decoder.set_last_layer(self.layer_dim)

    def forward(self, x):
        data_list = list(x.values())
        if len(data_list[0].size()) > 2:
            data = torch.flatten(data_list[0], start_dim=1)
        else:
            data = data_list[0]

        for id in range(1, len(data_list)):
            data = torch.concat((data, data_list[id]), dim=-1)

        z = torch.Tensor

        mean, logvar = self.encoder(data)
        std = torch.exp(torch.mul(logvar, 0.5))
    
        # Reparameterization trick
        dist = torch.distributions.Normal(self.mean, self.std)
        eps = dist.sample(mean.shape).to(self.device)

        z = torch.add(mean, torch.mul(std, eps))
        tmp = self.decoder(z)
        
        self.kld += - self.kld_scale * torch.sum(1 + logvar - mean.pow(2) - std.pow(2))

        x_hat = dict.fromkeys(x.keys())
        for id, key in enumerate(x_hat.keys()):
            x_hat[key] = tmp[:, self.modality_dims[id]:self.modality_dims[id]+self.modality_dims[id+1]]
            if key == 'image':
                x_hat[key] = torch.reshape(x_hat[key], (x_hat[key].size(dim=0), 1, 28, 28))

        return x_hat, z
    
    
    def loss(self, x, x_hat):
        loss_function = nn.MSELoss().to(self.device)
        recon_losses =  dict.fromkeys(x.keys())

        for key in x.keys():
            recon_losses[key] = self.scales[key] * loss_function(x_hat[key], x[key])

        recon_loss = 0
        for value in recon_losses.values():
            recon_loss += value
        
        self.kld = self.kld / len(list(x.values())[0])
        elbo = self.kld + recon_loss

        if recon_losses.get('trajectory') is None:
            recon_losses['trajectory'] = 0.
        elif recon_losses.get('image') is None:
            recon_losses['image'] = 0.

        loss_dict = Counter({'Total loss': elbo, 'KLD': self.kld, 'Img recon loss': recon_losses['image'], 'Traj recon loss': recon_losses['trajectory']})
        self.kld = 0.
        return elbo, loss_dict
        
