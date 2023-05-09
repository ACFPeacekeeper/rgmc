import torch
import torch.nn as nn

from typing import Tuple
from input_transformations.adversarial_attack import AdversarialAttack

class FGSM(AdversarialAttack):
    def __init__(self, device, model: nn.Module, eps=8/255) -> None:
        super().__init__("FGSM", model)
        self.eps = eps
        self.device = device

    def example_generation(self, x: Tuple[torch.Tensor], y: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        loss = nn.MSELoss().cuda(self.device)
        x_adv = [torch.empty(x[0].size())]*len(x)
        for idx, (x_sample, y_sample) in enumerate(zip(x, y)):
            x_sample = x_sample.detach().clone().to(self.device)
            y_sample = y_sample.detach().clone().to(self.device)

            x_sample.requires_grad = True
            results = self.model(x_sample)

            cost = loss(results[0], y_sample)

            grad = torch.autograd.grad(cost, x_sample, retain_graph=False, create_graph=False)[0]

            x_adv_sample = x_sample + self.eps * grad.sign()
            x_adv[idx] = torch.clam(x_adv_sample, 0, 1)

        return x_adv
    
    def __repr__(self) -> str:
        return self.__class__.__name__ + '(epsilon={0})'.format(self.eps)
