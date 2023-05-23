import os
import torch
import wandb
import tracemalloc

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import StrMethodFormatter

def save_results(m_path, config, device, loss_dict=None):
    if loss_dict is not None:
        with open(os.path.join(m_path, "results", config['stage'], config['model_out'] + ".txt"), 'a') as file:
            for key, value in loss_dict.items():
                wandb.log({f'epoch_{key}': value})
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

def save_train_results(m_path, config, loss_list_dict, bt_loss_dict):
    for idx, (key, values) in enumerate(loss_list_dict.items()):
        plt.figure(idx, figsize=(20, 20))
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.10f}'))
        plt.plot(range(config['epochs']), values, label='loss values')
        plt.xlabel("Epoch")
        plt.ylabel(key)
        plt.title(f'{key} per epoch')
        plt.legend()
        plt.savefig(os.path.join(m_path, "results", config['stage'], config['model_out'] + f'_{key}.png'))
        plt.close(idx)

    for idx, (key, values) in enumerate(bt_loss_dict.items()):
        plt.figure(idx+len(list(loss_list_dict.keys())), figsize=(20, 20))
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.10f}'))
        plt.plot(range(len(values)), values, label='loss values')
        plt.xlabel("Batch")
        plt.ylabel(key)
        plt.title(f'{key} per batch')
        plt.legend()
        plt.savefig(os.path.join(m_path, "results", config['stage'], config['model_out'] + f'_bl_{key}.png'))
        plt.close(idx+len(list(loss_list_dict.keys())))

    with open(os.path.join(m_path, "results", config['stage'], config['model_out'] + ".txt"), 'a') as file:
        print('Average epoch results:')
        file.write('Average epoch results:\n')
        for key, values in loss_list_dict.items():
            print(f'{key}: {np.mean(values)}')
            file.write(f'- {key}: {np.mean(values)}\n')
            
    return


def save_final_metrics(m_path, config, loss_dict, first_loss_dict, loss_list_dict):
    keys = list(loss_dict.keys())
    loss_dict = [tensor.item() if isinstance(tensor, torch.Tensor) else tensor for tensor in list(loss_dict.values())]
    first_loss_dict = [tensor.item() if isinstance(tensor, torch.Tensor) else tensor for tensor in list(first_loss_dict.values())]
    X_axis = np.arange(len(keys))
    plt.figure(figsize=(20, 10))
    plt.bar(X_axis - 0.2, loss_dict, width=0.4, label='Loss values', color='purple')
    if 'clf' in config['model_out']:
        plt.bar(X_axis + 0.2, [abs(fl_i - ls_i) for fl_i, ls_i in zip(first_loss_dict, loss_dict)], width=0.4, label='Loss improvement', color='powderblue')
    else:
        plt.bar(X_axis + 0.2, [fl_i - ls_i for fl_i, ls_i in zip(first_loss_dict, loss_dict)], width=0.4, label='Loss improvement', color='powderblue')
    for X_value, (key, value) in zip(X_axis, loss_list_dict.items()):
        plt.plot(X_value , np.mean(np.asarray([value_i for value_i in value])), marker="o", markersize=10, label=f'Avg {key.lower()}')
    plt.xticks(X_axis, keys)
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title("Loss values and improvements of model")
    plt.legend()
    plt.savefig(os.path.join(m_path, "results", config['stage'], config['model_out'] + '_metrics.png'))
    plt.close()
    return


def save_preds(m_path, config, preds, labels):
    print(f'Prediction count: {preds}')
    with open(os.path.join(m_path, "results", config['stage'], config['model_out'] + ".txt"), 'a') as file:
        file.write(f'Prediction count: {preds}\n')
    if preds is not None and labels is not None:
        X_axis = np.arange(len(preds))
        diff = [x if x >= 0 else -x for x in torch.bincount(labels).detach().cpu().numpy() - preds]
        plt.figure(figsize=(20, 10))
        plt.bar(X_axis - 0.2, preds, width=0.4, label='Predictions', color='blue')
        plt.bar(X_axis + 0.2, diff, width=0.4, label='Incorrect predictions', color='red')
        plt.xlabel('MNIST Digit')
        plt.ylabel('Number of examples')
        plt.title('Classifier predictions and number of incorrect predictions')
        plt.legend()
        plt.savefig(os.path.join(m_path, "results", config['stage'], config['model_out'] + '_predictions.png'))
        plt.close()
    return