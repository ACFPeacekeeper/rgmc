import torch
import torch.nn as nn

from collections import Counter
from architectures.dae_networks import Encoder, Decoder

class DAE(nn.Module):
    def __init__(self, name, latent_dim, device, exclude_modality, scales, noise_factor=0.3, test=False):
        super(DAE, self).__init__()
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
        self.test = test
        self.noise_factor = noise_factor

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

    def add_noise(self, x):
        x_noisy = []*len(x)
        for id, modality in enumerate(x):
            x_noisy[id] = torch.clip(torch.add(modality, torch.mul(torch.randn_like(modality), self.noise_factor)), 0., 1.)
        return x_noisy

    def forward(self, x):
        data_list = list(x.values())
        if not self.test:
            data_list = self.add_noise(data_list)

        if len(data_list[0].size()) > 2:
            data = torch.flatten(data_list[0], start_dim=1)
        else:
            data = data_list[0]

        for id in range(1, len(data_list)):
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
        loss_function = nn.MSELoss().to(self.device)
        recon_losses =  dict.fromkeys(x.keys())

        for key in x.keys():
            recon_losses[key] = self.scales[key] * loss_function(x_hat[key], x[key])

        recon_loss = 0
        for value in recon_losses.values():
            recon_loss += value

        if recon_losses.get('trajectory') is None:
            recon_losses['trajectory'] = 0.
        elif recon_losses.get('image') is None:
            recon_losses['image'] = 0.

        loss_dict = Counter({'Total loss': recon_loss, 'Img recon loss': recon_losses['image'], 'Traj recon loss': recon_losses['trajectory']})
        return recon_loss, loss_dict
