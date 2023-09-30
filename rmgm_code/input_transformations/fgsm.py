import torch
import torch.nn as nn
import torch.nn.functional as F

from input_transformations.adversarial_attack import AdversarialAttack

# Code adapted from https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/torchattacks/attacks/fgsm.py
class FGSM(AdversarialAttack):
    def __init__(self, name, model, device, target_modality, eps=8/255, targeted=False):
        super().__init__("FGSM", name, model, device, target_modality, targeted)
        self.eps = eps


    def __call__(self, x, y):
        x = x.clone().detach().to(self.device)
        for key in x.keys():
            x[key].requires_grad = True
            #x[key] = torch.unsqueeze(x[key], dim=0)
            
        result, _ = self.model(x)

        if y is not None:
            y = y.clone().detach().to(self.device)
            y = y.float()
            if y.dim() == 0:
                y = torch.unsqueeze(F.one_hot(y, result.size(dim=-1)), dim=0)
            
            loss = nn.CrossEntropyLoss().to(self.device)
            cost = loss(result, y)
        else:
            loss = nn.MSELoss().to(self.device)
            cost = loss(result[self.target_modality], x[self.target_modality])

        if self.targeted:
            cost = (-1) * cost

        grad = torch.autograd.grad(cost, x[self.target_modality], retain_graph=False, create_graph=False)[0]
        
        x_adv = torch.empty(x[self.target_modality].size())
        x_adv = torch.clamp(x[self.target_modality] + self.eps * grad.sign(), torch.min(x[self.target_modality]), torch.max(x[self.target_modality]))
        
        x[self.target_modality] = x_adv.clone().detach().to(self.device)
        #for key in x.keys():
            #x[key] = torch.squeeze(x[key], dim=0)

        return x
    

    def __repr__(self):
        return self.__class__.__name__ + '(epsilon={0})'.format(self.eps)