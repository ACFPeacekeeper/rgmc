from torch import autograd, clamp
from torch.nn import BCEWithLogitsLoss
from input_transformations.adversarial_attack import AdversarialAttack

import torch.nn.functional as F

# Code adapted from https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/torchattacks/attacks/bim.py
class BIM(AdversarialAttack):
    def __init__(self, model, device, target_modality, eps=0.1, alpha=2/255, steps=10, targeted=False, attack_mode="default"):
        super().__init__("BIM", model, device, target_modality, targeted, attack_mode)
        self.eps = eps
        self.alpha = alpha
        if steps == 0:
            self.steps = int(min(eps * 255 + 4, 1.25 * eps * 255))
        else:
            self.steps = steps

    def __call__(self, x, y):
        x_adv = dict.fromkeys(x)
        for key in x.keys():
            x_adv[key] = x[key].clone().detach().to(self.device)
            x_adv[key].requires_grad = True

        y = y.clone().detach().to(self.device)

        loss = BCEWithLogitsLoss()

        original_x = x_adv[self.target_modality].clone().detach()

        for _ in range(self.steps):
            for key in x.keys():
                x_adv[key].requires_grad = True
                
            model_output, _ = self.model(x_adv)
            if y.dim() == 1:
                y = F.one_hot(y, model_output.size(dim=-1)).float()

            cost = loss(model_output, y)
            grad = autograd.grad(cost, x_adv[self.target_modality], retain_graph=False, create_graph=False)[0]

            x_adv[self.target_modality] = x[self.target_modality] + self.alpha * grad.sign()
            a = clamp(original_x - self.eps, min=0)
            b = (x_adv[self.target_modality] >= a).float() * x_adv[self.target_modality] + (x_adv[self.target_modality] < a).float() * a
            c = (b > original_x + self.eps).float() * (original_x + self.eps) + (b <= original_x + self.eps).float() * b
            x_adv[self.target_modality] = clamp(c, max=1).detach()

        return x_adv

