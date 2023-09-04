import torch

from torch.utils.data import Dataset

class MultimodalDataset(Dataset):
    def __init__(self, name, dataset_dir, device, download=False, exclude_modality='none', target_modality='none', train=True, transform=None, adv_attack=None):
        super().__init__()
        if download:
            self._download()

        self.name = name
        self.device = device
        self.dataset_dir = dataset_dir
        self.exclude_modality = exclude_modality
        self.transform = transform
        self.adv_attack = adv_attack
        self.target_modality = target_modality
        self.dataset = {}
        self.dataset_len = 0
        self.labels = None
        self.modalities = None
        self._load_data(train)


    def _download(self):
        raise NotImplementedError
    
    def _load_data(self, train):
        raise NotImplementedError
    
    def _get_name(self):
        return self.name
    
    def _get_modalities(self):
        return self.modalities
    
    def _set_transform(self, transform):
        self.transform = transform
    
    def _set_adv_attack(self, adv_attack):
        self.adv_attack = adv_attack

    def __len__(self):
        return self.dataset_len
    
    def __getitem__(self, index):
        data = dict.fromkeys(self.dataset.keys())
        for key in data.keys():
            data[key] = self.dataset[key][index].type(torch.cuda.FloatTensor)

        if self.labels is not None:
            labels = self.labels[index]

        if self.transform is not None:
            data = self.transform(data)

        if self.adv_attack is not None:
            if self.labels is not None:
                data = self.adv_attack(data, labels)
            else:
                data = self.adv_attack(data, data)

        return data, labels