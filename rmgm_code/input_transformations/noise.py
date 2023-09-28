class Noise(object):
    def __init__(self, name, device, mean=0., std=1.):
        self.name = name
        self.mean = mean
        self.std = std

        self.device = device

    def __call__(self, x, y):
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
