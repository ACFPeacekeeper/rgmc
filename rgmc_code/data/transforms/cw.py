import torch
import torch.nn as nn

from .adversarial_attack import AdversarialAttack


# Code adapted from https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/torchattacks/attacks/cw.py#L123
class CW(AdversarialAttack):
    def __init__(self, name, model, device, target_modality, c_val=0.1, kappa=10, learning_rate=0.001, steps=10, targeted=False, attack_mode="default"):
        super().__init__(name, model, device, target_modality, targeted, attack_mode)
        self.c_val = c_val
        self.steps = steps
        self.kappa = kappa
        self.learning_rate = learning_rate

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
            loss = nn.MSELoss(reduction='none').to(self.device)

        # w = torch.zeros_like(images).detach() # Requires 2x times
        w = self._inverse_tanh_space(adv_x[self.target_modality]).detach().to(self.device)
        w.requires_grad = True

        best_adv_mod = adv_x[self.target_modality].clone().detach().to(self.device)
        best_L2_dist = 1e10 * torch.ones((len(adv_x[self.target_modality]))).to(self.device)
        prev_cost = 1e10
        x_dim = len(adv_x[self.target_modality].shape)

        flatten = nn.Flatten()
        optimizer = torch.optim.Adam([w], lr=self.learning_rate)
        for step in range(self.steps):
            # Get adversarial modality
            adv_x[self.target_modality] = self._tanh_space(w)

            # Calculate loss values
            current_L2 = loss(flatten(adv_x[self.target_modality], flatten(x[self.target_modality])))
            L2_loss = current_L2.sum()
            result, _ = self.model(adv_x)
            if self.targeted:
                f_loss = self._f_function(result, target_labels).sum()
            else:
                f_loss = self._f_function(result, y).sum()

            cost = L2_loss + self.c_val * f_loss
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            preds = torch.argmax(result.detach(), 1)
            if self.targeted:
                condition = (preds == target_labels).float()
            else:
                condition = (preds != y).float()

            # Filter out images that get either correct predictions or non-decreasing loss,
            # i.e., only images that are both misclassified and loss-decreasing are left
            mask = condition * (best_L2_dist > current_L2.detach())
            best_L2_dist = mask * current_L2.detach() + (1 - mask) * best_adv_mod
            mask = mask.view([-1] + [1] * (x_dim - 1))
            best_adv_mod = mask * adv_x[self.target_modality].detach() + (1 - mask) * best_adv_mod

            # Early stop when loss does not converge.
            if step % max(self.steps//10, 1) == 0:
                if cost.item() > prev_cost:
                    adv_x[self.target_modality] = best_adv_mod
                    return adv_x
                prev_cost = cost.item()

        adv_x[self.target_modality] = best_adv_mod
        return adv_x

    def _tanh_space(self, x):
        return 1/2*(torch.tanh(x) + 1)    

    def _inverse_tanh_space(self, x):
        # atanh is defined in the range -1 to 1
        return self._atanh(torch.clamp(x*2-1, min=-1, max=1))

    def _atanh(self, x):
        return 0.5*torch.log((1+x)/(1-x))

    def _f_function(self, outputs, labels):
        one_hot_labels = torch.eye(outputs.shape[1]).to(self.device)[labels]

        # find the max logit other than the target class
        other = torch.max((1 - one_hot_labels) * outputs, dim=1)[0]

        # get the target class's logit
        real = torch.max(one_hot_labels * outputs, dim=1)[0]
        if self.targeted:
            return torch.clamp((other - real), min = -self.kappa)
        else:
            return torch.clamp((real - other), min = -self.kappa)