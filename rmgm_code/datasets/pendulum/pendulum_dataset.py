import os, sys
from subprocess import call
from ..multimodal_dataset import *

class PendulumDataset(MultimodalDataset):
    def __init__(self, dataset_dir, device, download=False, exclude_modality='none', target_modality='none',  train=True, transform=None, adv_attack=None):
        super().__init__(dataset_dir, device, download, exclude_modality, target_modality, train, transform, adv_attack)
        self.properties = {}

    def _download(self):
        call("./download_pendulum_dataset.sh", shell=True)
        return

    def _load_data(self, train):
        if train:
            data_path = os.path.join(self.dataset_dir, "train_pendulum_dataset_samples20000_stack2_freq440.0_vel20.0_rec['LEFT_BOTTOM', 'RIGHT_BOTTOM', 'MIDDLE_TOP'].pt")
        else:
            data_path = os.path.join(self.dataset_dir, "test_pendulum_dataset_samples2000_stack2_freq440.0_vel20.0_rec['LEFT_BOTTOM', 'RIGHT_BOTTOM', 'MIDDLE_TOP'].pt")

        data = torch.load(data_path)
        self.dataset_len = len(data[0])
        self.labels = data[2] 
        self.dataset = {"freq": data[6], "amp": data[7], "done_t": data[3]}
        if self.exclude_modality == 'image':
            self.dataset["audio_t"] = data[1]
            self.dataset["audio_t+1"] = data[5]
        elif self.exclude_modality == 'audio':
            self.dataset["image_t"] = data[0]
            self.dataset["image_t+1"] = data[4]
        else:
            self.dataset["image_t"] = data[0]
            self.dataset["audio_t"] = data[1]
            self.dataset["image_t+1"] = data[4]
            self.dataset["audio_t+1"] = data[5]

        return