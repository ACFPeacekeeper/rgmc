import torch
import collections
import torch.nn as nn
import torch.nn.functional as F


class MMClassifier(nn.Module):
    def __init__(self, latent_dimension, model, exclude_modality):
        super(MMClassifier, self).__init__()
        self.model = model
        self.exclude_modality = exclude_modality
        self.latent_dimension = latent_dimension

        self.proj1 = nn.Linear(latent_dimension, latent_dimension)
        self.proj2 = nn.Linear(latent_dimension, latent_dimension)
        self.classifier = nn.Linear(latent_dimension, 1)

    def set_modalities(self, exclude_modality):
        self.exclude_modality = exclude_modality
        self.model.set_modalities(exclude_modality)

    def set_latent_dim(self, latent_dim):
        self.latent_dimension = latent_dim
        self.model.set_latent_dim(latent_dim)

    def set_perturbation(self, perturbation):
        self.model.set_perturbation(perturbation)

    
    def forward(self, x, sample=True):
        if 'gmc' in self.model.name:
            z = self.model.encode(x, sample)
        else:
            _, z = self.model(x, sample)
        encodings = self.proj2(F.relu(self.proj1(z)))
        encodings += z
        return self.classifier(F.relu(encodings)), z

    def loss(self, y_preds, labels):
        y_preds = y_preds.unsqueeze(-1)
        batch_size = labels.size()[0]
        loss_function = nn.L1Loss()
        loss = loss_function(y_preds, labels)
        accuracy = 0.
        for pred, label in zip(y_preds, labels):
            num_pred = torch.argmax(torch.exp(pred))
            accuracy += int(num_pred == label)

        accuracy = accuracy / batch_size
        loss_dict = collections.Counter({'nll_loss': loss, 'accuracy': accuracy})
        return loss, loss_dict
    
    def training_step(self, x, labels):
        classification, _ = self.forward(x, sample=True)
        loss, loss_dict = self.loss(classification, labels)
        return loss, loss_dict
    
    def validation_step(self, x, labels):
        return self.training_step(x, labels)
        