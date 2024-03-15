import os
import torch

from subprocess import run
from ..multimodal_dataset import MultimodalDataset

class MoseiDataset(MultimodalDataset):
    def __init__(self, dataset_dir, device, download=False, exclude_modality='none', target_modality='none', train=True, transform=None, adv_attack=None):
        super().__init__(dataset_dir, device, download, exclude_modality, target_modality, train, transform, adv_attack)

    @staticmethod
    def _download():
        run([os.path.join(os.getcwd(), "datasets", "mosei", "download_mosei_dataset.sh"), "bash"], shell=True)
        return
    
    def _load_data(self, train):   
        if train:
            data_path = os.path.join(self.dataset_dir, "mosei_train.dt")
            data = torch.load(data_path)
            val_data_path = os.path.join(self.dataset_dir, "mosei_valid.dt")
            val_data = torch.load(val_data_path)
            self.dataset = {'vision': torch.concat((data.vision, val_data.vision)).to(self.device), 'text': torch.concat((data.text, val_data.text)).to(self.device)}
            self.labels = torch.concat((data.labels, val_data.labels)).to(self.device)
            
        else:
            data_path = os.path.join(self.dataset_dir, "mosei_test.dt")
            data = torch.load(data_path)
            self.dataset = {'vision': data.vision.to(self.device), 'text': data.text.to(self.device)}
            self.labels = data.labels.to(self.device)

        self.dataset_len = len(self.labels)

        if self.exclude_modality != 'none' and self.exclude_modality is not None:
            self.dataset[self.exclude_modality] = torch.full(self.dataset[self.exclude_modality], -1).to(self.device)

        for mod in ['vision', 'text']:
            if mod != self.exclude_modality:
                self.dataset[mod] = (self.dataset[mod] - torch.min(self.dataset[mod])) / (torch.max(self.dataset[mod]) - torch.min(self.dataset[mod]))

        return