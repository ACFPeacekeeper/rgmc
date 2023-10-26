from torch import autograd, clamp
from torch.nn import CrossEntropyLoss
from input_transformations.adversarial_attack import AdversarialAttack

# Code adapted from https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/torchattacks/attacks/bim.py
class BIM(AdversarialAttack):
    def __init__(self, model, device, target_modality, eps=8/255, alpha=2/255, steps=10, targeted=False, attack_mode="default"):
        super().__init__("BIM", model, device, target_modality, targeted, attack_mode)
        self.eps = eps
        self.alpha = alpha
        if steps == 0:
            self.steps = int(min(eps * 255 + 4, 1.25 * eps * 255))
        else:
            self.steps = steps

    def forward(self, x, y):
        x = x.clone().detach().to(self.device)
        y = y.clone().detach().to(self.device)

        loss = CrossEntropyLoss()

        original_x = x[self.target_modality].clone().detach()

        for _ in range(self.steps):
            for key in x.keys():
                x[key].requires_grad = True
                
            model_output = self.model(x)

            cost = loss(model_output[self.target_modality], y)
            grad = autograd.grad(cost, x, retain_graph=False, create_graph=False)[0]

            adv_x = x[self.target_modality] + self.alpha * grad.sign()
            a = clamp(original_x - self.eps, min=0)
            b = (adv_x >= a).float() * adv_x + (adv_x < a).float() * a
            c = (b > original_x + self.eps).float() * (original_x + self.eps) + (b <= original_x + self.eps).float() * b
            x[self.target_modality] = clamp(c, max=1).detach()

        return x

