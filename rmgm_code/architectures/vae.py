import torch
import torch.nn as nn

from collections import Counter
from architectures.vae_networks import Encoder, Decoder

class VAE(nn.Module):
    def __init__(self, latent_dim, device, beta=0.5):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        
        self.device = device
        self.beta = beta
        self.kld = 0.

    def forward(self, batch):
        z = [torch.Tensor]*len(batch)
        x_hat = [(torch.Tensor, torch.Tensor)]*len(batch)
        for idx, x in enumerate(batch):
            mean, logvar = self.encoder(x)
            std = torch.exp(logvar/2)
        
            # Reparameterization trick
            dist = torch.distributions.Normal(0, 1)
            eps = dist.sample(mean.shape).to(self.device)

            z[idx] = mean + std * eps
            x_hat[idx] = self.decoder(z[idx])
            self.kld += - self.beta * (1 + logvar - (mean)**2 - (std)**2).sum()

        return x_hat, z
    
    
    def loss(self, batch, recons, scales):
        recon_loss = nn.MSELoss().cuda(self.device)
        img_recon_loss = 0.
        traj_recon_loss = 0.

        for x, x_hat in zip(batch, recons):
            img_recon_loss += recon_loss(x_hat[0], x[0])
            traj_recon_loss += recon_loss(x_hat[1], x[1])

        img_recon_loss /= len(batch)
        traj_recon_loss /= len(batch)
        
        elbo = self.kld + (scales['Image recon scale'] * img_recon_loss + scales['Trajectory recon scale'] * traj_recon_loss)

        loss_dict = Counter({'Total loss': elbo, 'KLD': self.kld, 'Img recon loss': img_recon_loss, 'Traj recon loss': traj_recon_loss})
        self.kld = 0.
        return elbo, loss_dict
        
