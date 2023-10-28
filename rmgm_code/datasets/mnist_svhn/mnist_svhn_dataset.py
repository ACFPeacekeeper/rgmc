import os
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from ..multimodal_dataset import *
from torchvision import datasets, transforms

# Adapted from https://github.com/iffsid/mmvae
class MnistSvhnDataset(MultimodalDataset):
    def __init__(self, name, dataset_dir, device, download=False, exclude_modality='none', target_modality='none', train=True, transform=None, adv_attack=None, max_d = 10000, dm=30):
        super().__init__(name, dataset_dir, device, download, exclude_modality, target_modality, train, transform, adv_attack)
        self.modalities = ["mnist", "svhn"]
        self.max_d = max_d  # maximum number of datapoints per class
        self.dm = dm        # data multiplier: random permutations to match 
        
    def _download(self):
        # Get the individual datasets
        tx = transforms.ToTensor()
        train_mnist = datasets.MNIST(os.path.join("datasets", "mnist_svhn"), train=True, download=True, transform=tx)
        test_mnist = datasets.MNIST(os.path.join("datasets", "mnist_svhn"), train=False, download=True, transform=tx)
        train_svhn = datasets.SVHN(os.path.join("datasets", "mnist_svhn"), split='train', download=True, transform=tx)
        test_svhn = datasets.SVHN(os.path.join("datasets", "mnist_svhn"), split='test', download=True, transform=tx)
        # SVHN labels need extra work
        train_svhn.labels = torch.LongTensor(train_svhn.labels.squeeze().astype(int)) % 10
        test_svhn.labels = torch.LongTensor(test_svhn.labels.squeeze().astype(int)) % 10

        train_dict = {"mnist": [], "svhn": [], "labels": []}
        svhn_dict = {"0": [], "1": [], "2": [], "3": [], "4": [], "5": [], "6": [], "7": [], "8": [], "9": []}
        print("Exporting svhn train set...")
        for feats, label in tqdm(zip(train_svhn.data, train_svhn.labels), total=len(train_svhn)):
            svhn_dict[str(label.item())].append(feats)

        mnist_dict = {"0": [], "1": [], "2": [], "3": [], "4": [], "5": [], "6": [], "7": [], "8": [], "9": []}
        print("Exporting mnist train set...")
        for feats, label in tqdm(zip(train_mnist.data, train_mnist.targets), total=len(train_mnist)):
            mnist_dict[str(label.item())].append(feats)
        
        print("Combining training datasets...")
        for dig in tqdm(svhn_dict.keys()):
            for mnist_feats, svhn_feats in zip(mnist_dict[str(dig)], svhn_dict[str(dig)]):
                train_dict["mnist"].append(mnist_feats[None, :])
                train_dict["svhn"].append(torch.from_numpy(svhn_feats))
                train_dict["labels"].append(torch.tensor(int(dig)))

        train_dict["mnist"] = torch.stack(train_dict["mnist"])
        train_dict["svhn"] = torch.stack(train_dict["svhn"])
        train_dict["labels"] = torch.stack(train_dict["labels"])

        test_dict = {"mnist": [], "svhn": [], "labels": []}
        svhn_dict = {"0": [], "1": [], "2": [], "3": [], "4": [], "5": [], "6": [], "7": [], "8": [], "9": []}
        print("Exporting svhn test set...")
        for feats, label in tqdm(zip(test_svhn.data, test_svhn.labels), total=len(test_svhn)):
            svhn_dict[str(label.item())].append(feats)

        mnist_dict = {"0": [], "1": [], "2": [], "3": [], "4": [], "5": [], "6": [], "7": [], "8": [], "9": []}
        print("Exporting mnist test set...")
        for feats, label in tqdm(zip(test_mnist.data, test_mnist.targets), total=len(test_mnist)):
            mnist_dict[str(label.item())].append(feats)
        
        print("Combining test datasets...")
        for dig in tqdm(svhn_dict.keys()):
            for mnist_feats, svhn_feats in zip(mnist_dict[str(dig)], svhn_dict[str(dig)]):
                test_dict["mnist"].append(mnist_feats[None, :])
                test_dict["svhn"].append(torch.from_numpy(svhn_feats))
                test_dict["labels"].append(torch.tensor(int(dig)))

        test_dict["mnist"] = torch.stack(test_dict["mnist"])
        test_dict["svhn"] = torch.stack(test_dict["svhn"])
        test_dict["labels"] = torch.stack(test_dict["labels"])
        
        torch.save(train_dict, os.path.join("datasets", "mnist_svhn", 'mnist_svhn_train.pt'))
        torch.save(test_dict, os.path.join("datasets", "mnist_svhn", 'mnist_svhn_test.pt'))
        return
    
    def _load_data(self, train):
        if train:
            data_path = os.path.join(self.dataset_dir, "mnist_svhn_train.pt")
        else:
            data_path = os.path.join(self.dataset_dir, "mnist_svhn_test.pt")

        data = torch.load(data_path)
        self.dataset_len = len(data["labels"])

        # Normalize datasets
        data['mnist'] = (data['mnist'] - torch.min(data['mnist'])) / (torch.max(data['mnist']) - torch.min(data['mnist']))
        data['svhn'] = (data['svhn'] - torch.min(data['svhn'])) / (torch.max(data['svhn']) - torch.min(data['svhn']))

        if self.exclude_modality == 'mnist':
            self.dataset = {'mnist': torch.full(data["mnist"].size(), -1).to(self.device), 'svhn': data["svhn"].to(self.device)}
        elif self.exclude_modality == 'svhn':
            self.dataset = {'mnist': data["mnist"].to(self.device), 'svhn': torch.full(data["svhn"].size(), -1).to(self.device)}
        else:
            self.dataset = {'mnist': data["mnist"].to(self.device), 'svhn': data["svhn"].to(self.device)}

        self.labels = data["labels"].to(self.device)
        return
    
    def _show_dataset_label_distribution(self):
        label_dict = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0}

        for data_set in ['train', 'test']:
            data_path = os.path.join(self.dataset_dir, f"mnist_svhn_{data_set}.pt")

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
            ax.set_title(f"MNIST-SVHN {data_set} set digit labels")
            ax.yaxis.grid(True)
            metrics_bar = ax.bar(X_axis, label_dict.values(), width=1, label="Loss values", align='center', ecolor='black', capsize=10)
            ax.bar_label(metrics_bar)
            fig.legend()
            fig.savefig(os.path.join(self.dataset_dir, f'ms_{data_set}.png'))
            plt.close()
        return