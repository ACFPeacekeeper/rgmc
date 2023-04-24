import torch
import torch.nn as nn

from adversarial_attack import AdversarialAttack

class FGSM(AdversarialAttack):
    def __init__(self, model: nn.Module, eps=8/255) -> None:
        super().__init__("FGSM", model)
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.detach().clone()
        y = y.detach().clone()

        loss = nn.CrossEntropyLoss()

        x.requires_grad = True
        y_hat = self.model(x)

        cost = loss(y_hat, y)

        grad = torch.autograd.grad(cost, x, retain_graph=False, create_graph=False)[0]

        x_adv = x + self.eps * grad.sign()
        x_adv = torch.clam(x_adv, 0, 1).detach()

        return x_adv