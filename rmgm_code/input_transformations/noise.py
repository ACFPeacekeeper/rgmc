class Noise(object):
    def __init__(self, device, mean=0., std=1.) -> None:
        self.mean = mean
        self.std = std

        self.device = device

    def add_noise(self, x):
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
