class AdversarialAttack(object):
    def __init__(self, name, model):
        self.name = name
        self.model = model

    def __call__(self, x, y=None):
        raise NotImplementedError
    
    def __repr__(self):
        raise NotImplementedError