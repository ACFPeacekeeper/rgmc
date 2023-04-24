class AdversarialAttack(object):
    def __init__(self, name, model):
        self.name = name
        self.model = model

    def forward(self, x, labels=None):
        raise NotImplementedError