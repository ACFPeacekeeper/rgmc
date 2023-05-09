from torch import nn
from torch.nn import functional as F

class Classifier(nn.Module):
    def __init__(self, latent_dim) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(latent_dim, 256),
            F.relu(),
            nn.Linear(256, 128),
            F.relu(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.layers(x)