import os
import torch
from subprocess import call
from ..MultimodalDataset import *

class PendulumDataset(MultimodalDataset):
    def __init__(self, dataset_dir, device, download=False, exclude_modality='none', target_modality='none',  train=True, transform=None, adv_attack=None):
        super().__init__(dataset_dir, device, download, exclude_modality, target_modality, train, transform, adv_attack)
        self.properties = {}

    def _download(self):
        call("./download_pendulum_dataset.sh")
        return

    def _load_data(self, train):
        if self.download:
            self._download
            
        return super()._load_data(self.dataset_dir, train)
        if train:
            data_path = os.path.join(self.dataset_dir, "train_pendulum_dataset_samples20000_stack2_freq440.0_vel20.0_rec['LEFT_BOTTOM', 'RIGHT_BOTTOM', 'MIDDLE_TOP'].pt")
        else:
            data_path = os.path.join(self.dataset_dir, "test_pendulum_dataset_samples2000_stack2_freq440.0_vel20.0_rec['LEFT_BOTTOM', 'RIGHT_BOTTOM', 'MIDDLE_TOP'].pt")

        data = torch.load(data_path)
        self.dataset = data

        self.properties = data[7].copy()
        