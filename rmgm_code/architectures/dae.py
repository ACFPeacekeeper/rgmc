import torch
import torch.nn as nn

from rmgm_code.architectures.dae_networks import Encoder, Decoder

class DAE(nn.Module):
    def __init__(self, latent_dim, noise_factor=0.3):
        super(DAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

        self.noise_factor = noise_factor

    def add_noise(self, x):
        img_noisy = x[0] + torch.randn_like(x[0]) * self.noise_factor
        traj_noisy = x[1] + torch.randn_like(x[1]) * self.noise_factor
        return [torch.clip(img_noisy, 0., 1.), torch.clip(traj_noisy, 0., 1.)]

    def forward(self, x):
        x_noisy = self.add_noise(x)
        z = self.encoder(x_noisy)
        x_hat = self.decoder(z)
        return z, x_hat
    
    def loss_mse(self, x, x_hat):
        loss = torch.nn.MSELoss()
        img_loss = loss(x_hat[0], x[0])
        traj_loss = loss(x_hat[1], x[1])
        total_loss = 0.5 * img_loss + 0.5 * traj_loss
        return total_loss

        
