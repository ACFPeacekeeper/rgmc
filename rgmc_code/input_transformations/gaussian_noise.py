import torch

from input_transformations.noise import Noise

class GaussianNoise(Noise):
    def __init__(self, device, target_modality=None, mean=0., std=1.):
        super().__init__("GaussianNoise", device, target_modality, mean, std)


    def __call__(self, x, y=None):
        for key in x.keys():
            if key == self.target_modality and self.std > 0:
                x[key] = torch.clamp(x[key] + torch.randn_like(x[key]) * self.std + self.mean, torch.min(x[key]), torch.max(x[key]))
            else:
                x[key] = x[key]
        return x


    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)