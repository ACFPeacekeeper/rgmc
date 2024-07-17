# Code adapted from https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/torchattacks/attack.py#L419
class AdversarialAttack(object):
    def __init__(self, name, model, device, target_modality, targeted=False, attack_mode="default"):
        self.name = name
        self.model = model
        self.device = device
        self.targeted = targeted
        self.attack_mode = attack_mode
        self.supported_modes = ['default']
        self.target_modality = target_modality

    def __call__(self, x, y):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

    def _set_mode_targeted(self, mode, quiet):
        if "targeted" not in self.supported_modes:
            raise ValueError("Targeted mode is not supported.")
        else:
            self.targeted = True
            self.attack_mode = mode
            if not quiet:
                print("Attack mode is changed to '%s'." % mode)

    def _get_target_label(self, inputs, labels=None):
        if self.attack_mode == 'targeted(label)':
            target_labels = labels
        else:
            if self._target_map_function is None:
                raise ValueError('target_map_function is not initialized by set_mode_targeted.')
            else:
                target_labels = self._target_map_function(inputs, labels)
        
        return target_labels
    
    def _set_target_modality(self, target_modality):
        self.target_modality = target_modality