import os
from subprocess import call
from ..MultimodalDataset import *

class MHDDataset(MultimodalDataset):
    def __init__(self, dataset_dir, device, download=False, exclude_modality='none', target_modality='none', train=True, transform=None, adv_attack=None):
        super().__init__(dataset_dir, device, download, exclude_modality, target_modality, train, transform, adv_attack)
        self.modalities = ["image", "trajectory"] 

    def _download(self):
        call("./download_mhd_dataset.sh", shell=True)
        return

    def _load_data(self, train):
        if train:
            data_path = os.path.join(self.dataset_dir, "mhd_train.pt")
        else:
            data_path = os.path.join(self.dataset_dir, "mhd_test.pt")
        
        data = torch.load(data_path)
        self.dataset_len = len(data[0])

        if self.exclude_modality == 'image':
            self.dataset = {'image': torch.full(data[1].size(), -1).to(self.device),'trajectory': data[2].to(self.device)}
        elif self.exclude_modality == 'trajectory':
            self.dataset = {'image': data[1].to(self.device), 'trajectory': torch.full(data[2].size(), -1).to(self.device)}
        else:
            self.dataset = {'image': data[1].to(self.device), 'trajectory': data[2].to(self.device)}

        self.labels = data[0].to(self.device)

        return