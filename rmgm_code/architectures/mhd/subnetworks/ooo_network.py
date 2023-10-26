import torch
import torch.nn as nn

class OddOneOutNetwork(nn.Module):
    def __init__(self, latent_dim, num_modalities, modalities, device):
        super().__init__()
        self.latent_dimension = latent_dim
        self.num_modalities = num_modalities
        self.modalities = modalities
        self.device = device
        self.embedder = nn.Sequential(
            nn.Conv1d(num_modalities, 64, 4, 2, 1),
            nn.GELU(),
            nn.Conv1d(64, 128, 4, 2, 1),
            nn.GELU(),
        )
        self.clf_fc = nn.Linear(2048, num_modalities + 1)
        self.classificator = nn.Softmax(dim=-1)

    def set_latent_dim(self, latent_dim):
        self.latent_dimension = latent_dim

    def forward(self, mod_representations):
        batch_size = mod_representations[0].size()[0]
        representations = mod_representations[0].view(batch_size, 1, self.latent_dimension)
        for mod_rep in mod_representations[1:]:
            mod_rep = mod_rep.view(batch_size, 1, self.latent_dimension)
            representations = torch.cat((representations, mod_rep), dim=1)
        h = self.embedder(representations)
        h = self.clf_fc(h.view(h.size(0), -1))
        classes = self.classificator(h)
        return classes