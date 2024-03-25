from datasets.mhd.mhd_dataset import MhdDataset
from datasets.mnist_svhn.mnist_svhn_dataset import MnistSvhnDataset
from datasets.mosei.mosei_dataset import MoseiDataset
from datasets.mosi.mosi_dataset import MosiDataset
from datasets.pendulum.pendulum_dataset import PendulumDataset

if __name__ == '__main__':
    MhdDataset._download()
    MnistSvhnDataset._download()
    MoseiDataset._download()
    MosiDataset._download()
    PendulumDataset._download()