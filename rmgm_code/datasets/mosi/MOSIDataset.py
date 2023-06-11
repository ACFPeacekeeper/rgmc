import os
import torch
import numpy as np
from subprocess import call
from ..MultimodalDataset import *

class MOSIDataset(MultimodalDataset):
    def __init__(self, dataset_dir, device, download=False, exclude_modality='none', train=True, transform=None, adv_attack=None, target_modality='none'):
        super().__init__(dataset_dir, device, download, exclude_modality, target_modality, train, transform, adv_attack)

    def _download(self):
        call("./download_mosi_dataset.sh")
        return
    
    def _load_data(self, train):     
        modalities = ['vision', 'text']
        if self.exclude_modality != 'none':
            modalities.remove(self.exclude_modality)

        if train:
            data_path = os.path.join(self.dataset_dir, "mosi_train.pt")
            data = torch.load(open(data_path, 'rb'))
            val_data_path = os.path.join(self.dataset_dir, "mosi_valid.pt")
            val_data = torch.load(open(val_data_path, 'rb'))
            for modal in modalities:
                self.dataset[modal] = torch.concat((data[modal], val_data[modal]), dim=0)
        else:
            data_path = os.path.join(self.dataset_dir, "mosi_test.pt")
            data = torch.load(open(data_path, 'rb'))
            for modal in modalities:
                self.dataset[modal] = torch.tensor(data['test'][modal].astype(np.float32))

        if train:
            self.labels = torch.concat((data['labels'], val_data['labels']), dim=0)
        else:
            self.labels = data['labels']

        return