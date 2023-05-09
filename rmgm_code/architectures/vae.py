import torch
import torch.nn as nn

from typing import Tuple
from collections import Counter
from architectures.vae_networks import Encoder, Decoder

class VAE(nn.Module):
    def __init__(self, latent_dim: int, device) -> None:
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        
        self.device = device

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        mean, logvar = self.encoder(x)
        std = torch.exp(logvar/2)
        
        # Reparameterization trick
        dist = torch.distributions.Normal(0, 1)
        eps = dist.sample(mean.shape).to(self.device)

        z = mean + std * eps
        
        x_hat = self.decoder(z)

        return z, mean, std, x_hat
    
    def loss(self, x: Tuple[torch.Tensor, torch.Tensor], z: torch.Tensor, mean: torch.Tensor, 
                  std: torch.Tensor, x_hat: Tuple[torch.Tensor, torch.Tensor], beta=0.5) -> Tuple[float, dict]:
        
        kld = beta * (1 + torch.log(std**2) - (mean)**2 - (std)**2).sum()

        recon_loss = torch.nn.MSELoss().cuda(self.device)

        img_recon_loss = recon_loss(x_hat[0], x[0])

        traj_recon_loss = recon_loss(x_hat[1], x[1])

        elbo = kld + (0.5 * img_recon_loss + 0.5 * traj_recon_loss)

        loss_dict = Counter({'ELBO': elbo, 'KLD': kld, 'Img recon loss': img_recon_loss, 'Traj recon loss': traj_recon_loss})

        return elbo, loss_dict
        
