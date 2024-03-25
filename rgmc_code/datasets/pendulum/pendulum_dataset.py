import os
import torch

from subprocess import run
from ..multimodal_dataset import MultimodalDataset

class PendulumDataset(MultimodalDataset):
    def __init__(self, dataset_dir, device, download=False, exclude_modality='none', target_modality='none', train=True, transform=None, adv_attack=None):
        super().__init__(dataset_dir, device, download, exclude_modality, target_modality, train, transform, adv_attack)

    @staticmethod
    def _download():
        dataset_dir = os.path.join(os.getcwd(), "datasets", "pendulum")
        run([os.path.join(dataset_dir, "download_pendulum_dataset.sh"), "bash"], shell=True)
        before_filenames = ["train_dataset_samples20000_stack2_freq440.0_vel20.0_rec[\'LEFT_BOTTOM\'\,\ \'RIGHT_BOTTOM\'\,\ \'MIDDLE_TOP\'].pt", "test_dataset_samples2000_stack2_freq440.0_vel20.0_rec[\'LEFT_BOTTOM\'\,\ \'RIGHT_BOTTOM\'\,\ \'MIDDLE_TOP\'].pt"]
        after_filenames = ["train_dataset_samples20000_stack2_freq440.0_vel20.0_rec.pt", "test_dataset_samples2000_stack2_freq440.0_vel20.0_rec.pt"]
        for name_before, name_after in zip(before_filenames, after_filenames):
            os.rename(name_before, name_after)
        return
    
    def _load_data(self, train):
        if train:
            data_path = os.path.join(self.dataset_dir, "train_dataset_samples20000_stack2_freq440.0_vel20.0_rec.pt")
        else:
            data_path = os.path.join(self.dataset_dir, "test_dataset_samples2000_stack2_freq440.0_vel20.0_rec.pt")
        
        data = torch.load(data_path)
        self.dataset_len = len(data[0])

        # Normalize datasets
        data[0] = (data[0] - torch.min(data[0])) / (torch.max(data[0]) - torch.min(data[0]))
        data[1] = (data[1] - torch.min(data[1])) / (torch.max(data[1]) - torch.min(data[1]))
        data[4] = (data[4] - torch.min(data[4])) / (torch.max(data[4]) - torch.min(data[4]))
        data[5] = (data[5] - torch.min(data[5])) / (torch.max(data[5]) - torch.min(data[5]))
        self.dataset = {
            'image_t': torch.full(data[0].size(), -1).to(self.device),      # size (n_samples x n_channels x npixels_width x npixels_height) = 20000||2000 x 2 x 60 x 60
            'audio_t': data[1].to(self.device),                             # size (n_samples x n_channels x n_microphones x sound_features[amplitude, frequency]) = 20000||2000 x 2 x 3 x 2
            'reward_t': data[2].to(self.device),                            # size (n_samples x 1) = 20000||2000 x 1 (float value)
            'done_t': data[3].to(self.device),                              # size (n_samples) = 20000||2000 (boolean value)
            'image_t++': torch.full(data[4].size(), -1).to(self.device),    # size (n_samples x n_channels x npixels_width x npixels_height) = 20000||2000 x 2 x 60 x 60
            'audio_t++': data[5].to(self.device),                           # size (n_samples x n_channels x n_microphones x sound_features[amplitude, frequency]) = 20000||2000 x 2 x 3 x 2
            'amplitude': data[7]['amplitude'],                              # [np.float64, np.float64]
            'frequency': data[7]['frequency']                               # [np.float64, np.float64]
            }

        if self.exclude_modality != 'none' and self.exclude_modality is not None:
            self.dataset[f'{self.exclude_modality}'] = torch.full(self.dataset[f'{self.exclude_modality}'].size(), -1).to(self.device)
            self.dataset[f'{self.exclude_modality}++'] = torch.full(self.dataset[f'{self.exclude_modality}++'].size(), -1).to(self.device)

        return