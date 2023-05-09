import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple
from input_transformations.adversarial_attack import AdversarialAttack

class FGSM(AdversarialAttack):
    def __init__(self, device, model: nn.Module, eps=8/255) -> None:
        super().__init__("FGSM", model)
        self.eps = eps
        self.device = device

    def example_generation(self, x: Tuple[torch.Tensor], y: Tuple[torch.Tensor], target_modality: str) -> Tuple[torch.Tensor]:
        loss = nn.MSELoss().cuda(self.device)
        if target_modality == 'image':
            x_adv = [torch.empty(len(x[0][0]))]*len(x)
            x_same = [torch.empty(len(x[0][1]))]*len(x)
        elif target_modality == 'trajectory':
            x_adv = [torch.empty(len(x[0][1]))]*len(x)
            x_same = [torch.empty(len(x[0][0]))]*len(x)
        else:
            raise ValueError

        
        for idx, (x_sample, y_sample) in enumerate(zip(x, y)):
            x_sample = [x_sample[0].detach().clone().to(self.device), x_sample[1].detach().clone().to(self.device)]
            
            if len(y_sample == 2):
                y_sample = [y_sample[0].detach().clone().to(self.device), y_sample[1].detach().clone().to(self.device)]
            else:
                y_sample = y_sample.detach().clone().to(self.device)

            if target_modality == 'image':
                x_sample[0].requires_grad = True
                x_same[idx] = x_sample[1]
            elif target_modality == 'trajectory':
                x_sample[1].requires_grad = True
                x_same[idx] = x_sample[0]

            if len(y_sample) == 2:
                results = self.model(x_sample)
                if target_modality == 'image':
                    cost = loss(results[0][0], y_sample[0])
                    grad = torch.autograd.grad(cost, x_sample[0], retain_graph=False, create_graph=False)[0]
                    x_adv_sample = x_sample[0] + self.eps * grad.sign()
                elif target_modality == 'trajectory':
                    cost = loss(results[0][1], y_sample[1])
                    grad = torch.autograd.grad(cost, x_sample[1], retain_graph=False, create_graph=False)[0]
                    x_adv_sample = x_sample[1] + self.eps * grad.sign()
            else:
                y_pred, _ = self.model(x_sample)
                one_hot_label = F.one_hot(y_sample, num_classes=len(y_pred))
                cost = F.nll_loss(y_pred, one_hot_label)
                grad = torch.autograd.grad(cost, y_sample, retain_graph=False, create_graph=False)[0]
                if target_modality == 'image':
                    x_adv_sample = x_sample[0] + self.eps * grad.sign()
                elif target_modality == 'trajectory':
                    x_adv_sample = x_sample[1] + self.eps * grad.sign()
                
            x_adv[idx] = torch.clamp(x_adv_sample, 0, 1)

        if target_modality == 'image':
            x = list(zip(x_adv, x_same))
        elif target_modality == 'trajectory':
            x = list(zip(x_same, x_adv))
        
        return x
    
    def __repr__(self) -> str:
        return self.__class__.__name__ + '(epsilon={0})'.format(self.eps)
