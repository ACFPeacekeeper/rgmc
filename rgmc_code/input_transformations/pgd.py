import torch
import torch.nn as nn
import torch.nn.functional as F

from input_transformations.adversarial_attack import AdversarialAttack

# Code adapted from https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/torchattacks/attacks/pgd.py
class PGD(AdversarialAttack):
    def __init__(self, model, device, target_modality, eps=0.1, alpha=2/255, steps=10, random_start=True, targeted=False, attack_mode="default"):
        super().__init__("PGD", model, device, target_modality, targeted, attack_mode)
        self.eps = eps
        self.steps = steps
        self.alpha = alpha
        self.random_start = random_start


    def __call__(self, x, y):
        adv_x = dict.fromkeys(x)
        for key in x.keys():
            adv_x[key] = x[key].clone().detach().to(self.device)

        if y is not None:
            y = y.clone().detach().to(self.device)
            loss = nn.BCEWithLogitsLoss().to(self.device)
            if self.targeted:
                target_labels = self._get_target_label(adv_x, y)
        else:
            loss = nn.MSELoss().to(self.device)

        if self.random_start:
            # Starting at a uniformly random point
            adv_x[self.target_modality] = adv_x[self.target_modality] + torch.empty_like(adv_x[self.target_modality]).uniform_(-self.eps, self.eps)
            adv_x[self.target_modality] = torch.clamp(adv_x[self.target_modality], min=0, max=1).detach()

        for _ in range(self.steps):
            for key in x.keys():
                adv_x[key].requires_grad = True

            result, _ = self.model(adv_x)
            if y is not None:
                if y.dim() == 1:
                    y = F.one_hot(y, result.size(dim=-1)).float()
                if self.targeted:
                    cost = (-1) * loss(result, target_labels)
                else:
                    cost = loss(result, y)
            else:
                cost = loss(result[self.target_modality], adv_x[self.target_modality])

            grad = torch.autograd.grad(cost, adv_x[self.target_modality], retain_graph=False, create_graph=False)[0]
            adv_x[self.target_modality] = adv_x[self.target_modality].detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_x[self.target_modality] - x[self.target_modality], min=-self.eps, max=self.eps)
            adv_x[self.target_modality] = torch.clamp(x[self.target_modality] + delta, min=0, max=1).detach()

        x[self.target_modality] = adv_x[self.target_modality].clone().detach().to(self.device)
        return x