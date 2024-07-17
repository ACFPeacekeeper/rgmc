import os
import torch
import subprocess

from ..multimodal_dataset import MultimodalDataset


class MosiDataset(MultimodalDataset):
    def __init__(self, dataset_dir, device, download=False, exclude_modality='none', target_modality='none', train=True, transform=None, adv_attack=None):
        super().__init__(dataset_dir, device, download, exclude_modality, target_modality, train, transform, adv_attack)

    @staticmethod
    def _download():
        dataset_dir = os.path.join(os.getcwd(), "datasets", "mosi")
        subprocess.run([os.path.join(dataset_dir, "download_mosi_dataset.sh"), "bash"], shell=True)
        data = torch.load(os.path.join(dataset_dir, "mosi_train.dt"))
        val_data = torch.load(os.path.join(dataset_dir, "mosi_valid.dt"))
        dataset = {'text': torch.concat((data.text, val_data.text)), 'audio': torch.concat((data.audio, val_data.audio)), 'vision': torch.concat((data.vision, val_data.vision)), 'labels': torch.concat((data.labels, val_data.labels))}
        torch.save(dataset, os.path.join(dataset_dir, "mosi_train.pt"))
        data = torch.load(os.path.join(dataset_dir, "mosi_test.dt"))
        dataset = {'text': data.text, 'audio': data.audio, 'vision': data.vision, 'labels': data.labels}
        torch.save(dataset, os.path.join(dataset_dir, "mosi_test.pt"))
    
    def _load_data(self, train):     
        if train:
            data_path = os.path.join(self.dataset_dir, "mosi_train.pt")
            
        else:
            data_path = os.path.join(self.dataset_dir, "mosi_test.pt")

        data = torch.load(data_path)
        self.dataset = {
            'text': data['text'].to(self.device), 
            'audio': data['audio'].to(self.device), 
            'vision': data['vision'].to(self.device)
        }
        self.labels = data['labels'].to(self.device)
        self.dataset_len = len(self.labels)
        for mod in ['text', 'audio', 'vision']:
            if mod != self.exclude_modality:
                self.dataset[mod] = (self.dataset[mod] - torch.min(self.dataset[mod])) / (torch.max(self.dataset[mod]) - torch.min(self.dataset[mod]))

        if self.exclude_modality != 'none' and self.exclude_modality is not None:
            self.dataset[self.exclude_modality] = torch.full(self.dataset[self.exclude_modality], -1).to(self.device)