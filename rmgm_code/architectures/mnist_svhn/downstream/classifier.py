import torch
from torch import nn
from torch.nn import functional as F

from collections import Counter

class MSClassifier(nn.Module):
    def __init__(self, latent_dimension, model, exclude_modality):
        super(MSClassifier, self).__init__()
        self.model = model
        self.exclude_modality = exclude_modality
        self.latent_dimension = latent_dimension
        self.fc1 = nn.Linear(latent_dimension, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def set_modalities(self, exclude_modality):
        self.exclude_modality = exclude_modality
        self.model.set_modalities(exclude_modality)

    def set_latent_dim(self, latent_dim):
        self.latent_dimension = latent_dim
        self.model.set_latent_dim(latent_dim)

    def forward(self, x, sample=False):
        if 'gmc' in self.model.name:
            z = self.model.encode(x, sample)
        else:
            _, z = self.model(x, sample)
        encoding = self.fc1(z)
        encoding = F.relu(encoding)
        encoding = self.fc2(encoding)
        encoding = F.relu(encoding)
        encoding = self.fc3(encoding)
        classification = F.log_softmax(encoding, dim=-1)
        return classification, z
    
    def loss(self, y_preds, labels):
        batch_size = labels.size()[0]
        loss_function = nn.CrossEntropyLoss()
        loss = loss_function(y_preds, labels)
        accuracy = 0.
        for pred, label in zip(y_preds, labels):
            num_pred = torch.argmax(torch.exp(pred))
            accuracy += int(num_pred == label)

        accuracy = accuracy / batch_size
        loss_dict = Counter({'nll_loss': loss, 'accuracy': accuracy})
        return loss, loss_dict
    
    def training_step(self, x, labels):
        classification, _ = self.forward(x, sample=False)
        loss, loss_dict = self.loss(classification, labels)
        return loss, loss_dict
    
    def validation_step(self, x, labels):
        return self.training_step(x, labels)