import torch
import torch.nn as nn
import torch.nn.functional as F

class OddOneOutNetwork(nn.Module):
    def __init__(self, common_dim, latent_dimension, num_modalities):
        super().__init__()
        self.common_dim = common_dim
        self.latent_dimension = latent_dimension
        self.fc1 = nn.Linear(latent_dimension, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_modalities + 1)

    def set_latent_dim(self, latent_dim):
        self.latent_dimension = latent_dim

    def forward(self, mod_representations):
        return

    def loss(self, perturbed_preds, clean_pred):
        preds = torch.concat((perturbed_preds, clean_pred), dim=0)
        loss = - torch.mean(torch.log(preds))
        return loss, {"odd_one_out_loss:": loss}