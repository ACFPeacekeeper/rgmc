from torch import nn
from torch.nn import functional as F

class Classifier(nn.Module):
    def __init__(self, latent_dim, model):
        super().__init__()
        self.model = model
        self.layers = nn.Sequential(
            nn.Linear(latent_dim, 256),
            F.relu(),
            nn.Linear(256, 128),
            F.relu(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        encoding = self.model(x)
        classification = F.log_softmax(self.layers(encoding[0]))
        return classification, encoding
    
    def loss(self, y_pred, label):
        one_hot_label = F.one_hot(label, num_classes=len(y_pred))
        return F.nll_loss(y_pred, one_hot_label)