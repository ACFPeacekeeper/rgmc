import os
import math
import torch
import tracemalloc

import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from matplotlib.ticker import StrMethodFormatter
from torch.utils.data import DataLoader, random_split


def save_epoch_results(m_path, config, device, runtime, batch_number, loss_dict=None):
    print(f'Runtime: {runtime} sec')
    with open(os.path.join(m_path, "results", config['stage'], config['model_out'] + ".txt"), 'a') as file:
        file.write(f'- Runtime: {runtime} sec\n')
    
    if loss_dict is not None:
        with open(os.path.join(m_path, "results", config['stage'], config['model_out'] + ".txt"), 'a') as file:
            for key, value in loss_dict.items():
                value = value / batch_number
                print(f'{key}: {value}')
                file.write(f'- {key}: {value}\n')
    
    print('Current RAM usage: %f GB'%(tracemalloc.get_traced_memory()[0]/1024/1024/1024))
    print('Peak RAM usage: %f GB'%(tracemalloc.get_traced_memory()[1]/1024/1024/1024))
    if device.type == 'cuda':
        print("Torch CUDA memory allocated: %f GB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
        print("Torch CUDA memory reserved: %f GB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
        print("Torch CUDA max memory reserved: %f GB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
    with open(os.path.join(m_path, "results", config['stage'], config['model_out'] + ".txt"), 'a') as file:
        file.write('- Current RAM usage: %f GB\n'%(tracemalloc.get_traced_memory()[0]/1024/1024/1024))
        file.write('- Peak RAM usage: %f GB\n'%(tracemalloc.get_traced_memory()[1]/1024/1024/1024))
        if device.type == 'cuda':
            file.write("- Torch CUDA memory allocated: %f GB\n"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
            file.write("- Torch CUDA memory reserved: %f GB\n"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
            file.write("- Torch CUDA max memory reserved: %f GB\n"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))

    return

def plot_loss_graph(m_path, config, loss_list_dict, batch_number, label):
    keys = list(loss_list_dict.keys())
    loss_means = defaultdict(list)
    loss_stds = defaultdict(list)
    for idx, key in enumerate(keys):
        loss_values = np.array(loss_list_dict[key])
        epoch_means = np.mean(loss_values.reshape(-1, batch_number), axis=1)
        epoch_stds = np.std(loss_values.reshape(-1, batch_number), axis=1)
        loss_means[key] = epoch_means
        loss_stds[key] = epoch_stds
        plt.figure(idx, figsize=(20, 20))
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.10f}'))
        plt.plot(range(len(epoch_means)), epoch_means, label=label + " loss values")
        plt.fill_between(range(len(epoch_stds)), epoch_means-epoch_stds, epoch_means+epoch_stds, alpha=0.1)
        plt.xlabel("epoch")
        plt.ylabel(key)
        plt.title(f'{key} per epoch')
        plt.legend()
        plt.savefig(os.path.join(m_path, "results", config['stage'], config['model_out'] + f'_{label}_{key}.png'))
        plt.close(idx)

    return

def plot_metrics_bar(m_path, config, losses, val_losses=None):
    keys = list(losses.keys())
    X_axis = np.arange(len(keys))
    loss_means = [np.mean(loss) for loss in losses.values()]
    loss_stds = [np.std(loss) for loss in losses.values()]
    with open(os.path.join(m_path, "results", config["stage"], config["model_out"] + ".txt"), "a") as file:
        for key, mean in zip(keys, loss_means):
            print(f'{key}: {mean}')
            file.write(f'- {key}: {mean}\n')
    fig, ax = plt.subplots()
    fig.figsize=(20, 10)
    ax.set_xticks(X_axis)
    ax.set_xticklabels(keys)
    ax.set_title("Loss values of the model")
    ax.yaxis.grid(True)
    if val_losses is not None:
        train_bar = ax.bar(X_axis - 0.2, loss_means, yerr=loss_stds, width=0.4, label='Training loss values', align="center", alpha=0.5, ecolor='black', capsize=10)
        val_bar = ax.bar(X_axis + 0.2, [np.mean(val_loss) for val_loss in val_losses.values()], yerr=[np.std(val_loss) for val_loss in val_losses.values()], width=0.4, label='Validation loss values', align="center", alpha=0.5, ecolor='black', capsize=10)
        ax.bar_label(train_bar)
        ax.bar_label(val_bar)
    else:
        test_bar = ax.bar(X_axis, loss_means, yerr=loss_stds, width=0.4, label="Testing loss values", align='center', alpha=0.5, ecolor='black', capsize=10)
        ax.bar_label(test_bar)

    fig.legend()
    fig.savefig(os.path.join(m_path, "results", config['stage'], config['model_out'] + '_metrics.png'))
    plt.close()
    return

def save_train_results(m_path, config, train_losses, val_losses, dataset):
    train_set, val_set = random_split(dataset, [math.ceil(0.8 * dataset.dataset_len), math.floor(0.2 * dataset.dataset_len)])
    train_bnumber = len(iter(DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, drop_last=True)))
    val_bnumber = len(iter(DataLoader(val_set, batch_size=config['batch_size'], shuffle=True, drop_last=True)))
    plot_loss_graph(m_path, config, train_losses, train_bnumber, label="train")
    plot_loss_graph(m_path, config, val_losses, val_bnumber, label="validation")
    plot_metrics_bar(m_path, config, train_losses, val_losses)
    return

def save_test_results(m_path, config, loss_list_dict):
    plot_metrics_bar(m_path, config, loss_list_dict)
    return
