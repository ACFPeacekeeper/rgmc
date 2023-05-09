import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, latent_dim, exclude_modality, first_layer_dim):
        super(Encoder, self).__init__()
        self.first_layer = nn.Linear(first_layer_dim, 512)

        self.feature_extractor = nn.Sequential(
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
        )

        self.fc_mean = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        self.exclude_modality = exclude_modality
    
    def forward(self, x):
        if len(x) == 2:
            img = x[0]
            traj = x[1]
            feats = torch.concat((torch.flatten(img), traj))
        else:
            if self.exclude_modality == 'trajectory':
                feats = torch.flatten(x)
            elif self.exclude_modality == 'image':
                feats = x
            
        comb_feats = self.first_layer(feats)
        comb_feats = self.feature_extractor(comb_feats)

        mean = self.fc_mean(comb_feats)
        logvar = self.fc_logvar(comb_feats)
        return mean, logvar
    
class Decoder(nn.Module):
    def __init__(self, latent_dim, exclude_modality, last_layer_dim):
        super(Decoder, self).__init__()
        self.feature_reconstructor = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 512),
            nn.SiLU(),
        )

        self.last_layer = nn.Linear(512, last_layer_dim)
        self.exclude_modality = exclude_modality

    def forward(self, z):
        x_hat = self.feature_reconstructor(z)
        x_hat = self.last_layer(x_hat)
        if self.exclude_modality == 'trajectory':
            recon = (torch.reshape(x_hat, (1, 28, 28)))
        elif self.exclude_modality == 'image':
            recon = (x_hat)
        else:
            img_recon = x_hat[:28*28]
            img_recon = torch.reshape(img_recon, (1, 28, 28))
            traj_recon = x_hat[28*28:28*28+200]
            recon = (img_recon, traj_recon)
        return recon

