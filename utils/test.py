import os
import time
import torch
import collections
import tracemalloc
import matplotlib.pyplot as plt

from tqdm import tqdm
from utils.logger import save_test_results, save_trajectory


def run_inference(m_path, config, device, model, dataset):
    for modality in dataset._get_modalities():
        os.makedirs(os.path.join(m_path, "checkpoints", modality), exist_ok=True)

    print('Performing inference')
    with open(os.path.join(m_path, "results", config['path_model'] + ".txt"), 'a') as file:
        file.write('Performing inference:\n')

    dataloader = iter(torch.utils.data.DataLoader(dataset, batch_size=1))
    counter = 0
    tracemalloc.start()
    inference_start = time.time()
    for idx, (batch_feats, batch_labels) in enumerate(tqdm(dataloader, total=len(dataloader))):
        if config['checkpoint'] != 0 and counter % config['checkpoint'] == 0: 
            _, x_hat = model.inference(batch_feats, batch_labels)
            label = int(batch_labels[0])
            for modality in batch_feats.keys():
                if modality == 'image' or modality == 'mnist':
                    if "adversarial_attack" not in config or config["target_modality"] == "mnist" or config["target_modality"] == "image":
                        plt.imsave(os.path.join("checkpoints", modality, config['model_out'] + f'_{idx}_{label}_orig.png'), torch.squeeze(batch_feats[modality]).detach().clone().cpu())
                        plt.imsave(os.path.join("checkpoints", modality, config['model_out'] + f'_{idx}_{label}_recon.png'), torch.squeeze(x_hat[modality]).detach().clone().cpu())
                elif (modality == 'trajectory' and "adversarial_attack" not in config) or config["target_modality"] == "trajectory":
                    save_trajectory(m_path, os.path.join("checkpoints", "trajectory", config['model_out'] + f'_{idx}_{label}_orig.png'), batch_feats['trajectory'])
                    save_trajectory(m_path, os.path.join("checkpoints", "trajectory", config['model_out'] + f'_{idx}_{label}_recon.png'), x_hat['trajectory'])
                elif (modality == 'svhn' and "adversarial_attack" not in config) or config["target_modality"] == "svhn":
                    batch_feats['svhn'] = torch.squeeze(batch_feats['svhn']).permute(1, 2, 0).detach().clone().cpu().numpy()
                    x_hat['svhn'] = torch.squeeze(x_hat['svhn']).permute(1, 2, 0).detach().clone().cpu().numpy()
                    plt.imsave(os.path.join("checkpoints", "svhn", config['model_out'] + f'_{idx}_{label}_orig.png'), batch_feats['svhn'])
                    plt.imsave(os.path.join("checkpoints", "svhn", config['model_out'] + f'_{idx}_{label}_recon.png'), x_hat['svhn'])

        counter += 1

    inference_stop = time.time()
    tracemalloc.stop()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    print(f'Total runtime: {inference_stop - inference_start} sec')
    with open(os.path.join(m_path, "results", config['path_model'] + ".txt"), 'a') as file:
        file.write(f'- Total runtime: {inference_stop - inference_start} sec\n')


def run_test(m_path, config, device, model, dataset):
    dataloader = iter(torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True))
    test_bnumber = len(dataloader)
    loss_dict = collections.Counter(dict.fromkeys(dataset.dataset.keys(), 0.))
    tracemalloc.start()
    test_start = time.time()
    for batch_feats, batch_labels in tqdm(dataloader, total=test_bnumber):
        _, batch_loss_dict = model.validation_step(batch_feats, batch_labels)

        loss_dict = loss_dict + batch_loss_dict

    for key in loss_dict.keys():
        loss_dict[key] = [loss_dict[key] / test_bnumber]

    test_end = time.time()
    tracemalloc.stop()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    print(f'- Total runtime: {test_end - test_start} sec')
    with open(os.path.join(m_path, "results", config['stage'], config['model_out'] + ".txt"), 'a') as file:
        file.write(f'- Total runtime: {test_end - test_end} sec\n')

    save_test_results(m_path, config, loss_dict)