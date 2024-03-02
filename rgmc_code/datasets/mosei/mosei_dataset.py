import os
from subprocess import run
from ..multimodal_dataset import *

class MoseiDataset(MultimodalDataset):
    def __init__(self, dataset_dir, device, download=False, exclude_modality='none', target_modality='none', train=True, transform=None, adv_attack=None):
        super().__init__(dataset_dir, device, download, exclude_modality, target_modality, train, transform, adv_attack)
        self.meta = None

    @staticmethod
    def _download():
        run([os.path.join(os.getcwd(), "datasets", "mosei", "download_mosei_dataset.sh"), "bash"], shell=True)
        return
    
    def _load_data(self, train):   
        modalities = ['vision', 'text']
        if self.exclude_modality != 'none':
            modalities.remove(self.exclude_modality)

        if train:
            data_path = os.path.join(self.dataset_dir, "mosei_train.pt")
            data = torch.load(data_path)
            val_data_path = os.path.join(self.dataset_dir, "mosei_valid.pt")
            val_data = torch.load(val_data_path)
            for modal in modalities:
                self.dataset[modal] = torch.concat((data[modal], val_data[modal]), dim=0)
        else:
            data_path = os.path.join(self.dataset_dir, "mosei_test.pt")
            data = torch.load(data_path)
            for modal in modalities:
                self.dataset[modal] = data[modal]

        if train:
            self.labels = torch.concat((data['labels'], val_data['labels']), dim=0)
        else:
            self.labels = data['labels']

        return