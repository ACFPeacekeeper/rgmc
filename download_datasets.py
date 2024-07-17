from data.datasets import MhdDataset, MnistSvhnDataset, MoseiDataset, MosiDataset, PendulumDataset

if __name__ == '__main__':
    MhdDataset._download()
    MnistSvhnDataset._download()
    MoseiDataset._download()
    MosiDataset._download()
    PendulumDataset._download()