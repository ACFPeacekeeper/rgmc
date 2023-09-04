import os
from torchvision import datasets, transforms
from ..MultimodalDataset import *

# Adapted from https://github.com/iffsid/mmvae
class MNISTSVHNDataset(MultimodalDataset):
    def __init__(self, name, dataset_dir, device, download=False, exclude_modality='none', target_modality='none', train=True, transform=None, adv_attack=None, max_d = 10000, dm=30):
        super().__init__(name, dataset_dir, device, download, exclude_modality, target_modality, train, transform, adv_attack)
        self.modalities = ["mnist", "svhn"]
        self.max_d = max_d  # maximum number of datapoints per class
        self.dm = dm        # data multiplier: random permutations to match 

    def _rand_match_on_idx(self, l1, idx1, l2, idx2, max_d=10000, dm=10):
        _idx1, _idx2 = [], []
        for l in l1.unique():  # assuming both have same idxs
            l_idx1, l_idx2 = idx1[l1 == l], idx2[l2 == l]
            n = min(l_idx1.size(0), l_idx2.size(0), max_d)
            l_idx1, l_idx2 = l_idx1[:n], l_idx2[:n]
            for _ in range(dm):
                _idx1.append(l_idx1[torch.randperm(n)])
                _idx2.append(l_idx2[torch.randperm(n)])
        return torch.cat(_idx1), torch.cat(_idx2)
    
    def _download(self):
         # get the individual datasets
        tx = transforms.ToTensor()
        train_mnist = datasets.MNIST('../data', train=True, download=True, transform=tx)
        test_mnist = datasets.MNIST('../data', train=False, download=True, transform=tx)
        train_svhn = datasets.SVHN('../data', split='train', download=True, transform=tx)
        test_svhn = datasets.SVHN('../data', split='test', download=True, transform=tx)
        # svhn labels need extra work
        train_svhn.labels = torch.LongTensor(train_svhn.labels.squeeze().astype(int)) % 10
        test_svhn.labels = torch.LongTensor(test_svhn.labels.squeeze().astype(int)) % 10

        mnist_l, mnist_li = train_mnist.targets.sort()
        svhn_l, svhn_li = train_svhn.labels.sort()
        idx1, idx2 = self._rand_match_on_idx(mnist_l, mnist_li, svhn_l, svhn_li, max_d=self.max_d, dm=self.dm)
        print('len train idx:', len(idx1), len(idx2))
        torch.save(idx1, '../data/train-ms-mnist-idx.pt')
        torch.save(idx2, '../data/train-ms-svhn-idx.pt')

        mnist_l, mnist_li = test_mnist.targets.sort()
        svhn_l, svhn_li = test_svhn.labels.sort()
        idx1, idx2 = self._rand_match_on_idx(mnist_l, mnist_li, svhn_l, svhn_li, max_d=self.max_d, dm=self.dm)
        print('len test idx:', len(idx1), len(idx2))
        torch.save(idx1, '../data/test-ms-mnist-idx.pt')
        torch.save(idx2, '../data/test-ms-svhn-idx.pt')
        return
    
    def _load_data(self, train):
        if train:
            data_path = os.path.join(self.dataset_dir, "data", "")
        return