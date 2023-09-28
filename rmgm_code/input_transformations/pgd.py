import torch
import torch.nn as nn
import torch.nn.functional as F

from input_transformations.adversarial_attack import AdversarialAttack

class PGD(AdversarialAttack):
    def __init__(self, name, model, device, target_modality, eps=8/255, alpha=2/255, steps=10, random_start=True):
        super().__init__(name, model)
        self.eps = eps
        self.steps = steps
        self.alpha = alpha
        self.device = device
        self.random_start = random_start
        self.target_modality = target_modality

    def forward(self, x, y):
        adv_x = x.clone().detach().to(self.device)
        if self.random_start:
            # Starting at a uniformly random point
            adv_x[self.target_modality] = adv_x[self.target_modality] + torch.empty_like(adv_x[self.target_modality]).uniform_(-self.eps, self.eps)
            adv_x[self.target_modality] = torch.clamp(adv_x[self.target_modality], min=0, max=1).detach()

        if y is not None:
            y = y.clone().detach().to(self.device)
            y = y.float()
            loss = nn.CrossEntropyLoss().to(self.device)
        else:
            loss = nn.MSELoss().to(self.device)

        for _ in range(self.steps):
            for key in x.keys():
                adv_x[key].requires_grad = True

            result, _ = self.model(x)
            if y is not None:
                if y.dim() == 0:
                    y = torch.unsqueeze(F.one_hot(y, result.size(dim=-1)), dim=0)

                cost = loss(result, y)
            else:
                cost = loss(result, x)

            grad = torch.autograd.grad(cost, adv_x[self.target_modality], retain_graph=False, create_graph=False)[0]
            x[self.target_modality] = x[self.target_modality].detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_x[self.target_modality] - x[self.target_modality], min=-self.eps, max=self.eps)
            adv_x[self.target_modality] = torch.clamp(x[self.target_modality] + delta, min=0, max=1).detach()

        x = adv_x
        return x

            