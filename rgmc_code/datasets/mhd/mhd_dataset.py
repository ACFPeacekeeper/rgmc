import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from subprocess import run
from ..multimodal_dataset import MultimodalDataset

class MhdDataset(MultimodalDataset):
    def __init__(self, dataset_dir, device, download=False, exclude_modality='none', target_modality='none', train=True, transform=None, adv_attack=None):
        super().__init__(dataset_dir, device, download, exclude_modality, target_modality, train, transform, adv_attack)
        self.modalities = ["image", "trajectory"] 

    @staticmethod
    def _download():
        run([os.path.join(os.getcwd(), "datasets", "mhd", "download_mhd_dataset.sh"), "bash"], shell=True)
        return

    def _load_data(self, train):
        if train:
            data_path = os.path.join(self.dataset_dir, "mhd_train.pt")
        else:
            data_path = os.path.join(self.dataset_dir, "mhd_test.pt")
        
        print(data_path)
        data = list(torch.load(data_path))
        self.dataset_len = len(data[0])

        # Normalize datasets
        data[1] = (data[1] - torch.min(data[1])) / (torch.max(data[1]) - torch.min(data[1]))
        data[2] = (data[2] - torch.min(data[2])) / (torch.max(data[2]) - torch.min(data[2]))

        if self.exclude_modality == 'image':
            self.dataset = {'image': torch.full(data[1].size(), -1).to(self.device),'trajectory': data[2].to(self.device)}
        elif self.exclude_modality == 'trajectory':
            self.dataset = {'image': data[1].to(self.device), 'trajectory': torch.full(data[2].size(), -1).to(self.device)}
        else:
            self.dataset = {'image': data[1].to(self.device), 'trajectory': data[2].to(self.device)}

        self.labels = data[0].to(self.device)
        return
    
    def _show_dataset_label_distribution(self):
        label_dict = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0}
        
        for data_set in ['train', 'test']:
            data_path = os.path.join(self.dataset_dir, f"mhd_{data_set}.pt")

            data = torch.load(data_path)
            labels = data["labels"]
            dataset_len = len(labels)
            for label in labels:
                label_dict[str(label.item())] += 1

            print(f'Label count: {dataset_len}')
            print(label_dict)
            X_axis = np.arange(len(label_dict.keys()))
            fig, ax = plt.subplots()
            fig.figsize=(20, 10)
            ax.set_xticks(X_axis)
            ax.set_xticklabels(label_dict.keys())
            ax.set_title(f"MHD {data_set} set digit labels")
            ax.yaxis.grid(True)
            metrics_bar = ax.bar(X_axis, label_dict.values(), width=1, label="Loss values", align='center', ecolor='black', capsize=10)
            ax.bar_label(metrics_bar)
            fig.legend()
            fig.savefig(os.path.join(self.dataset_dir, f'mhd_{data_set}.png'))
            plt.close()
        return