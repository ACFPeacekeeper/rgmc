import torch
import torch.nn as nn
import torch.nn.functional as F

from input_transformations.adversarial_attack import AdversarialAttack

# Code adapted from https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/torchattacks/attacks/fgsm.py
class FGSM(AdversarialAttack):
    def __init__(self, model, device, target_modality, eps=8/255, targeted=False, attack_mode="default"):
        super().__init__("FGSM", model, device, target_modality, targeted, attack_mode)
        self.eps = eps


    def __call__(self, x, y=None):
        x_adv = dict.fromkeys(x)
        for key in x.keys():
            x_adv[key] = x[key].clone().detach().to(self.device)
            x_adv[key].requires_grad = True
            
        result, _ = self.model(x_adv)

        if y is not None:
            y = y.clone().detach().to(self.device)
            if y.dim() == 1:
                y = F.one_hot(y, result.size(dim=-1))
            
            loss = nn.BCEWithLogitsLoss().to(self.device)
            cost = loss(result, y.float())
        else:
            loss = nn.MSELoss().to(self.device)
            cost = loss(result[self.target_modality], x[self.target_modality])

        if self.targeted:
            cost = (-1) * cost

        grad = torch.autograd.grad(cost, x_adv[self.target_modality], retain_graph=False, create_graph=False)[0]
        x_adv[self.target_modality] = torch.clamp(x_adv[self.target_modality] + self.eps * grad.sign(), torch.min(x_adv[self.target_modality]), torch.max(x_adv[self.target_modality]))
        
        x[self.target_modality] = x_adv[self.target_modality].clone().detach().to(self.device)
        return x
    

    def __repr__(self):
        return self.__class__.__name__ + '(epsilon={0})'.format(self.eps)