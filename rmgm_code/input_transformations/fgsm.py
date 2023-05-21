import torch
import torch.nn as nn
import torch.nn.functional as F

from input_transformations.adversarial_attack import AdversarialAttack

class FGSM(AdversarialAttack):
    def __init__(self, device, model, target_modality, eps=8/255):
        super().__init__("FGSM", model)
        self.eps = eps
        self.device = device
        self.target_modality = target_modality

    def __call__(self, x, y=None):
        loss = nn.MSELoss().to(self.device)

        for key in x.keys():
            x[key].requires_grad = True
            
        x_adv = torch.empty(x[self.target_modality].size())
        print("Generating adversarial examples...")
        if y is not None:
            y.requires_grad = True
            classification, _, _ = self.model(x)
            cost = loss(classification, y)
        else:
            x_hat, _ = self.model(x)
            cost = loss(x_hat[self.target_modality], x[self.target_modality])

        
        grad = torch.autograd.grad(cost, x[self.target_modality], retain_graph=False, create_graph=False)[0]

        x_adv = torch.clamp(x[self.target_modality] + self.eps * grad.sign(), torch.min(x[self.target_modality]), torch.max(x[self.target_modality]))
        x[self.target_modality] = x_adv
        return x
    
    def __repr__(self):
        return self.__class__.__name__ + '(epsilon={0})'.format(self.eps)
