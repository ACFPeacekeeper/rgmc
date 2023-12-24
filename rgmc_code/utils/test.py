import os
import tracemalloc

from time import time
from tqdm import tqdm
from torch import cuda, squeeze
from collections import Counter
from matplotlib.pyplot import imsave
from torch.utils.data import DataLoader
from utils.logger import save_test_results, save_trajectory


def run_inference(m_path, config, device, model, dataset):
    for modality in dataset._get_modalities():
        os.makedirs(os.path.join(m_path, "checkpoints", modality), exist_ok=True)

    print('Performing inference')
    with open(os.path.join(m_path, "results", config['path_model'] + ".txt"), 'a') as file:
        file.write('Performing inference:\n')

    dataloader = iter(DataLoader(dataset, batch_size=1))
    tracemalloc.start()
    inference_start = time()
    counter = 0
    for idx, (batch_feats, batch_labels) in enumerate(tqdm(dataloader, total=len(dataloader))):
        if config['checkpoint'] != 0 and counter % config['checkpoint'] == 0: 
            _, x_hat = model.inference(batch_feats, batch_labels)
            label = int(batch_labels[0])
            for modality in batch_feats.keys():
                if modality == 'image' or modality == 'mnist':
                    if "adversarial_attack" not in config or config["target_modality"] == "mnist" or config["target_modality"] == "image":
                        imsave(os.path.join("checkpoints", modality, config['model_out'] + f'_{idx}_{label}_orig.png'), squeeze(batch_feats[modality]).detach().clone().cpu())
                        imsave(os.path.join("checkpoints", modality, config['model_out'] + f'_{idx}_{label}_recon.png'), squeeze(x_hat[modality]).detach().clone().cpu())
                elif (modality == 'trajectory' and "adversarial_attack" not in config) or config["target_modality"] == "trajectory":
                    save_trajectory(m_path, os.path.join("checkpoints", "trajectory", config['model_out'] + f'_{idx}_{label}_orig.png'), batch_feats['trajectory'])
                    save_trajectory(m_path, os.path.join("checkpoints", "trajectory", config['model_out'] + f'_{idx}_{label}_recon.png'), x_hat['trajectory'])
                elif (modality == 'svhn' and "adversarial_attack" not in config) or config["target_modality"] == "svhn":
                    batch_feats['svhn'] = squeeze(batch_feats['svhn']).permute(1, 2, 0).detach().clone().cpu().numpy()
                    x_hat['svhn'] = squeeze(x_hat['svhn']).permute(1, 2, 0).detach().clone().cpu().numpy()
                    imsave(os.path.join("checkpoints", "svhn", config['model_out'] + f'_{idx}_{label}_orig.png'), batch_feats['svhn'])
                    imsave(os.path.join("checkpoints", "svhn", config['model_out'] + f'_{idx}_{label}_recon.png'), x_hat['svhn'])

        counter += 1

    inference_stop = time()
    tracemalloc.stop()
    if device.type == 'cuda':
        cuda.empty_cache()

    print(f'Total runtime: {inference_stop - inference_start} sec')
    with open(os.path.join(m_path, "results", config['path_model'] + ".txt"), 'a') as file:
        file.write(f'- Total runtime: {inference_stop - inference_start} sec\n')
    
    return

def run_test(m_path, config, device, model, dataset):
    dataloader = iter(DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True))
    test_bnumber = len(dataloader)
    loss_dict = Counter(dict.fromkeys(dataset.dataset.keys(), 0.))
    tracemalloc.start()
    test_start = time()
    for batch_feats, batch_labels in tqdm(dataloader, total=test_bnumber):
        _, batch_loss_dict = model.validation_step(batch_feats, batch_labels)

        loss_dict = loss_dict + batch_loss_dict

    for key in loss_dict.keys():
        loss_dict[key] = [loss_dict[key] / test_bnumber]

    test_end = time()
    tracemalloc.stop()
    print(f'- Total runtime: {test_end - test_start} sec')
    with open(os.path.join(m_path, "results", config['stage'], config['model_out'] + ".txt"), 'a') as file:
        file.write(f'- Total runtime: {test_end - test_end} sec\n')

    save_test_results(m_path, config, loss_dict)
    if device.type == 'cuda':
        cuda.empty_cache()

    return