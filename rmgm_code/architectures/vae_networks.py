import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, latent_dim: int):
        super(Encoder, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(28 * 28 + 200, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
        )

        self.fc_mean = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
    
    def forward(self, x):
        img = x[0]
        traj = x[1]
        img = torch.flatten(img)
        comb_feats = self.feature_extractor(torch.concat((img, traj)))

        mean = self.fc_mean(comb_feats)
        logvar = self.fc_logvar(comb_feats)
        return mean, logvar
    
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.feature_reconstructor = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 512),
            nn.SiLU(),
            nn.Linear(512, 28 * 28 + 200)
        )

    def forward(self, z):
        x_hat = self.feature_reconstructor(z)
        img_recon = x_hat[:28*28]
        img_recon = torch.reshape(img_recon, (1, 28, 28))
        traj_recon = x_hat[28*28:28*28+200]
        return (img_recon, traj_recon)

