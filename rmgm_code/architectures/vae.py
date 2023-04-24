import torch
import torch.nn as nn

from rmgm_code.architectures.vae_networks import Encoder, Decoder

class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

        self.scale = torch.exp(nn.Parameter(torch.Tensor([0.0])))

    def forward(self, x):
        mean, logvar = self.encoder(x)
        std = torch.exp(torch.sqrt(logvar))
        
        dist = torch.distributions.Normal(0, 1)
        eps = dist.sample(mean.shape)

        z = mean + std * eps
        
        x_hat = self.decoder(z)

        return z, mean, std, x_hat
    
    def loss_elbo(self, x, z, mean, std, x_hat, beta=0.5):
        p = torch.distributions.Normal(torch.zeros_like(mean), torch.ones_like(std))
        q = torch.distributions.Normal(mean, std)

        log_q = q.log_prob(z)
        log_p = p.log_prob(z)

        kld = (log_q - log_p)
        kld = beta * kld.sum(-1)

        img_recon_loss = ((x[0] - x_hat[0])**2).sum()

        # TODO: Might use another loss for trajectory recon loss
        traj_recon_loss = ((x[1] - x_hat[1])**2).sum()

        elbo = (kld - (0.5 * img_recon_loss + 0.5 *traj_recon_loss))

        return elbo
        
