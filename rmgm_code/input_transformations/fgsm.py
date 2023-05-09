import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from input_transformations.adversarial_attack import AdversarialAttack

class FGSM(AdversarialAttack):
    def __init__(self, device, model, eps=8/255):
        super().__init__("FGSM", model)
        self.eps = eps
        self.device = device

    def example_generation(self, batch, targets, target_modality):
        loss = nn.MSELoss().cuda(self.device)
        if target_modality == 'image':
            x_adv = [torch.empty(len(batch[0][0]))]*len(batch)
            x_same = [torch.empty(len(batch[0][1]))]*len(batch)
        elif target_modality == 'trajectory':
            x_adv = [torch.empty(len(batch[0][1]))]*len(batch)
            x_same = [torch.empty(len(batch[0][0]))]*len(batch)
        else:
            raise ValueError
        
        if len(targets[0]) == 1:
            x_hat = self.model(batch)
        else:
            x_hat, _ = self.model(batch)

        
        for idx in range(len(batch)):
            if target_modality == 'image':
                batch[idx][0].requires_grad = True
                cost = loss(x_hat[idx][0], targets[idx][0])
                grad = torch.autograd.grad(cost, batch[idx][0], retain_graph=False, create_graph=False)[0]
                x_adv_sample = batch[idx][0] + self.eps * grad.sign()
                x_same[idx] = batch[idx][1]
            elif target_modality == 'trajectory':
                batch[idx][1].requires_grad = True
                cost = loss(x_hat[idx][1], targets[idx][1])
                grad = torch.autograd.grad(cost, batch[idx][1], retain_graph=False, create_graph=False)[0]
                x_adv_sample = batch[idx][1] + self.eps * grad.sign()
                x_same[idx] = batch[idx][0]
            else:
                one_hot_label = F.one_hot(targets[idx], num_classes=len(x_hat[idx]))
                cost = F.nll_loss(x_hat[idx], one_hot_label)
                if target_modality == 'image':
                    batch[idx][0].requires_grad = True
                    grad = torch.autograd.grad(cost, batch[idx][0], retain_graph=False, create_graph=False)[0]
                    x_adv_sample = batch[idx][0] + self.eps * grad.sign()
                    x_same[idx] = batch[idx][1]
                elif target_modality == 'trajectory':
                    batch[idx][1].requires_grad = True
                    grad = torch.autograd.grad(cost, batch[idx][1], retain_graph=False, create_graph=False)[0]
                    x_adv_sample = batch[idx][1] + self.eps * grad.sign()
                    x_same[idx] = batch[idx][0]
                
            x_adv[idx] = torch.clamp(x_adv_sample, 0, 1)

        if target_modality == 'image':
            x = list(zip(x_adv, x_same))
        elif target_modality == 'trajectory':
            x = list(zip(x_same, x_adv))
        
        return x
    
    def __repr__(self):
        return self.__class__.__name__ + '(epsilon={0})'.format(self.eps)
