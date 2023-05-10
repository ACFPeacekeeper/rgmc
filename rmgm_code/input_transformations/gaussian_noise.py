import torch

from input_transformations.noise import Noise

class GaussianNoise(Noise):
    def __init__(self, device, mean=0., std=1.):
        self.mean = mean
        self.std = std

        self.device = device

    def add_noise(self, x, target_modality):
        if len(x[0]) == 2:
            if target_modality == 'image':
                x_noisy = [torch.empty(x[0].size())]*len(x)
                x_same = [torch.empty(x[1].size())]*len(x)
                for idx, x_sample in enumerate(x):
                    x_noisy[idx] = x_sample[0] + torch.randn(x_sample[0].size()).to(self.device) * self.std + self.mean
                    x_same[idx] = x_sample[1]
                x_trans = list(zip(x_noisy, x_same))
            elif target_modality == 'traj':
                x_noisy = [torch.empty(len(x[0][1]))]*len(x)
                x_same = [torch.empty(len(x[0][0]))]*len(x)
                for x_sample in x:
                    x_noisy = x_sample[1] + torch.randn(x_sample[1].size()).to(self.device) * self.std + self.mean
                    x_same = x_sample[0]
                x_trans = list(zip(x_same, x_noisy))
            else:
                raise ValueError
        else:
            print(x[0].size())
            x_noisy = [torch.empty(x[0].size())]*len(x)
            for idx, x_sample in enumerate(x):
                
                x_noisy[idx] = x_sample[0] + torch.randn(x_sample[0].size()).to(self.device) * self.std + self.mean
            x_trans = list(x_noisy)
            print(x_trans[0].size())


        return x_trans

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
