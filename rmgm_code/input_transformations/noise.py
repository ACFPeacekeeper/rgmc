class Noise(object):
    def __init__(self, name, model, device, target_modality, mean=0., std=1.):
        self.std = std
        self.mean = mean
        self.name = name
        self.model = model
        self.device = device
        self.target_modality = target_modality


    def __call__(self, x, y):
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    
    def _set_target_modality(self, target_modality):
        self.target_modality = target_modality