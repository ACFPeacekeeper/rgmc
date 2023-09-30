import torch

from input_transformations.noise import Noise

class GaussianNoise(Noise):
    def __init__(self, model, device, target_modality, mean=0., std=1.):
        super().__init__("GaussianNoise", model, device, target_modality, mean, std)


    def __call__(self, x, y):
        for key in x.keys():
            if key == self.target_modality:
                x[key] = torch.clamp(x[key] + torch.randn(x[key].size()).to(self.device) * self.std + self.mean, torch.min(x[key]), torch.max(x[key]))
            else:
                x[key] = x[key]
        return x


    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)