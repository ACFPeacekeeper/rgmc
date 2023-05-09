import torch
import torch.nn as nn

from input_transformations.adversarial_attack import AdversarialAttack

class FGSM(AdversarialAttack):
    def __init__(self, device ,model: nn.Module, eps=8/255) -> None:
        super().__init__("FGSM", model)
        self.eps = eps
        self.device = device

    def example_generation(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.detach().clone().to(self.device)
        y = y.detach().clone().to(self.device)

        loss = nn.CrossEntropyLoss().cuda(self.device)

        x.requires_grad = True
        y_hat = self.model(x)

        cost = loss(y_hat, y)

        grad = torch.autograd.grad(cost, x, retain_graph=False, create_graph=False)[0]

        x_adv = x + self.eps * grad.sign()
        x_adv = torch.clam(x_adv, 0, 1)

        return x_adv
    
    def __repr__(self) -> str:
        return self.__class__.__name__ + '(epsilon={0})'.format(self.eps)
