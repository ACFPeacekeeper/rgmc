import torch
from torch import nn
from torch.nn import functional as F

class MNISTClassifier(nn.Module):
    def __init__(self, latent_dim, model):
        super().__init__()
        self.model = model
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x, sample=True):
        if self.model.name == 'GMC':
            encoding = self.model.encode(x)
            repr = encoding.detach().clone()
        else:
            repr, encoding = self.model(x, sample)
        encoding = self.fc1(encoding)
        encoding = F.relu(encoding)
        encoding = self.fc2(encoding)
        encoding = F.relu(encoding)
        encoding = self.fc3(encoding)
        classification = F.log_softmax(encoding, dim=-1)
        return classification, repr, encoding
    
    def loss(self, y_preds, labels):
        batch_size = labels.size()[0]
        num_preds = [0] * 10
        loss_function = nn.CrossEntropyLoss()
        loss = loss_function(y_preds, labels)
        accuracy = 0.
        for pred, label in zip(y_preds, labels):
            num_pred = torch.argmax(torch.exp(pred))
            accuracy += int(num_pred == label)
            num_preds[num_pred] += 1

        accuracy = accuracy / batch_size
        return loss, accuracy, num_preds