import torch
import torch.nn as nn

from collections import Counter
from architectures.dae_networks import Encoder, Decoder

class DAE(nn.Module):
    def __init__(self, latent_dim, device, noise_factor=0.3, test=False):
        super(DAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

        self.device = device

        self.test = test
        self.noise_factor = noise_factor

    def add_noise(self, x):
        img_noisy = x[0] + torch.randn_like(x[0]) * self.noise_factor
        traj_noisy = x[1] + torch.randn_like(x[1]) * self.noise_factor
        return [torch.clip(img_noisy, 0., 1.), torch.clip(traj_noisy, 0., 1.)]

    def forward(self, batch):
        z = [torch.Tensor]*len(batch)
        x_hat = [(torch.Tensor, torch.Tensor)]*len(batch)
        for idx, x in enumerate(batch):
            if not self.test:
                x = self.add_noise(x)

            z[idx] = self.encoder(x)
            x_hat[idx] = self.decoder(z[idx])

        return x_hat, z
    
    def loss(self, batch, recons, scales):
        loss = nn.MSELoss().cuda(self.device)
        img_loss = 0.
        traj_loss = 0.

        for x, x_hat in zip(batch, recons):
            img_loss += loss(x_hat[0], x[0])
            traj_loss += loss(x_hat[1], x[1])
        
        img_loss /= len(batch)
        traj_loss /= len(batch)

        total_loss = scales['Image recon scale'] * img_loss + scales['Trajectory recon scale'] * traj_loss

        loss_dict = Counter({'Total loss': total_loss, 'Img recon loss': img_loss, 'Traj recon loss': traj_loss})
        return total_loss, loss_dict
