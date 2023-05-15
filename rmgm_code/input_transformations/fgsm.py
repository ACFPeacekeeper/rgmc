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

    def example_generation(self, x, targets):
        loss = nn.MSELoss().to(self.device)

        for modality in x.keys():
            x[modality].requires_grad = True
            
        x_adv = torch.empty(targets.size())
        self.model.set_verbose(True)
        print("Generating adversarial examples...")
        x_hat, _ = self.model(x)
        self.model.set_verbose(False)

        cost = loss(x_hat[self.target_modality], targets)
        
        grad = torch.autograd.grad(cost, x[self.target_modality], retain_graph=False, create_graph=False)[0]

        x_adv = x[self.target_modality] + self.eps * grad.sign()

        return torch.clamp(x_adv, 0., 1.)
    
    def __repr__(self):
        return self.__class__.__name__ + '(epsilon={0})'.format(self.eps)
