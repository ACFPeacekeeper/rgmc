import os
import math
import torch
import wandb
import tracemalloc

import numpy as np
import matplotlib.pyplot as plt

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
                wandb.log({key: value})

    print('Current RAM usage: %f GB'%(tracemalloc.get_traced_memory()[0]/1024/1024/1024))
    print('Peak RAM usage: %f GB'%(tracemalloc.get_traced_memory()[1]/1024/1024/1024))
    if device.type == 'cuda':
        print("Torch CUDA memory allocated: %f GB"%(torch.cuda.memory_allocated(torch.cuda.current_device())/1024/1024/1024))
        print("Torch CUDA memory reserved: %f GB"%(torch.cuda.memory_reserved(torch.cuda.current_device())/1024/1024/1024))
        print("Torch CUDA max memory reserved: %f GB"%(torch.cuda.max_memory_reserved(torch.cuda.current_device())/1024/1024/1024))
    with open(os.path.join(m_path, "results", config['stage'], config['model_out'] + ".txt"), 'a') as file:
        file.write('- Current RAM usage: %f GB\n'%(tracemalloc.get_traced_memory()[0]/1024/1024/1024))
        file.write('- Peak RAM usage: %f GB\n'%(tracemalloc.get_traced_memory()[1]/1024/1024/1024))
        if device.type == 'cuda':
            file.write("- Torch CUDA memory allocated: %f GB\n"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
            file.write("- Torch CUDA memory reserved: %f GB\n"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
            file.write("- Torch CUDA max memory reserved: %f GB\n"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))

    return

def plot_loss_graph(m_path, config, loss_list_dict, batch_number, val_losses=None, val_bnumber=None):
    keys = list(loss_list_dict.keys())
    for idx, key in enumerate(keys):
        loss_values = np.array(loss_list_dict[key])
        epoch_means = np.mean(loss_values.reshape(-1, batch_number), axis=1)
        epoch_stds = np.std(loss_values.reshape(-1, batch_number), axis=1)
        plt.figure(idx, figsize=(20, 20))
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.10f}'))
        plt.plot(range(len(epoch_means)), epoch_means, label="loss values", color="blue", linewidth=2.0)
        plt.fill_between(range(len(epoch_stds)), epoch_means-epoch_stds, epoch_means+epoch_stds, color="blue", alpha=0.2)
        if val_losses is not None and val_bnumber is not None:
            valloss_values = np.array(val_losses[key])
            val_means = np.mean(valloss_values.reshape(-1, val_bnumber), axis=1)
            #val_stds = np.std(valloss_values.reshape(-1, val_bnumber), axis=1)
            plt.plot(range(len(val_means)), val_means, label="validation loss values", color="red", linewidth=2.0)
            #plt.fill_between(range(len(val_stds)), val_means-val_stds, val_means+val_stds, alpha=0.1)
        plt.xlabel("epoch")
        plt.ylabel(key)
        plt.title(f'{key} per epoch')
        plt.legend()
        plt.savefig(os.path.join(m_path, "results", config['stage'], config['model_out'] + f'_{key}.png'))
        plt.close(idx)

    return

def plot_metrics_bar(m_path, config, losses, val_losses=None):
    keys = list(losses.keys())
    X_axis = np.arange(len(keys))
    loss_means = [np.mean(loss) for loss in losses.values()]
    loss_yerr = [np.std(loss) for loss in losses.values()]
    loss_yerr = np.array(loss_yerr).T.tolist()
    with open(os.path.join(m_path, "results", config["stage"], config["model_out"] + ".txt"), "a") as file:
        for key in keys:
            loss = losses[key][-1]
            print(f'{key}: {loss}')
            file.write(f'- {key}: {loss}\n')
    fig, ax = plt.subplots()
    fig.figsize=(20, 10)
    ax.set_xticks(X_axis)
    ax.set_xticklabels(keys)
    ax.set_title("Loss values of the model")
    ax.yaxis.grid(True)
    if val_losses is not None:
        val_yerr = [np.std(loss) for loss in val_losses.values()]
        val_yerr = np.array(val_yerr).T.tolist()
        train_bar = ax.bar(X_axis - 0.2, loss_means, yerr=loss_yerr, width=0.4, label='Training loss values', align="center", alpha=0.5, ecolor='black', capsize=10)
        val_bar = ax.bar(X_axis + 0.2, [np.mean(val_loss) for val_loss in val_losses.values()], yerr=val_yerr, width=0.4, label='Validation loss values', align="center", alpha=0.5, ecolor='black', capsize=10)
        ax.bar_label(train_bar)
        ax.bar_label(val_bar)
    else:
        test_bar = ax.bar(X_axis, loss_means, yerr=loss_yerr, width=0.4, label="Testing loss values", align='center', alpha=0.5, ecolor='black', capsize=10)
        ax.bar_label(test_bar)

    fig.legend()
    fig.savefig(os.path.join(m_path, "results", config['stage'], config['model_out'] + '_metrics.png'))
    plt.close()
    return

def save_trajectory(path, traj_feats):
    traj_arr = traj_feats.cpu().numpy()
    traj_arr = traj_arr.reshape((traj_arr.size))
    x_arr = traj_arr[0::2]
    y_arr = traj_arr[1::2]
    plt.figure(figsize=(20, 20))
    plt.plot(x_arr, y_arr)
    plt.savefig(os.path.join("checkpoints", "trajectory", path))
    plt.close()     
    return

def save_train_results(m_path, config, train_losses, val_losses, dataset):
    train_set, val_set = random_split(dataset, [math.ceil(0.8 * dataset.dataset_len), math.floor(0.2 * dataset.dataset_len)])
    train_bnumber = len(iter(DataLoader(train_set, batch_size=config['batch_size'], drop_last=True)))
    val_bnumber = len(iter(DataLoader(val_set, batch_size=config['batch_size'], drop_last=True)))
    plot_loss_graph(m_path, config, train_losses, train_bnumber, val_losses, val_bnumber)
    plot_metrics_bar(m_path, config, train_losses, val_losses)
    return

def save_test_results(m_path, config, loss_list_dict):
    plot_metrics_bar(m_path, config, loss_list_dict)
    return
