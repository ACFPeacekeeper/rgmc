class AdversarialAttack(object):
    def __init__(self, name, model):
        self.name = name
        self.model = model

    def example_generation(self, x, y):
        raise NotImplementedError
    
    def __repr__(self):
        raise NotImplementedError