import os
import wandb
import torch
import tracemalloc
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import StrMethodFormatter


def save_config(file_path, config):
    with open(file_path, 'w') as file:
        for ckey, cval in config.items():
            if cval is not None:
                print(f'{ckey}: {cval}')
                file.write(f'{ckey}: {cval}\n')
    return

def save_epoch_results(m_path, config, device, runtime, loss_dict=None):
    print(f'- Runtime: {runtime} sec')
    with open(os.path.join(m_path, "results", config['stage'], config['model_out'] + ".txt"), 'a') as file:
        file.write(f'- Runtime: {runtime} sec\n')
    
    if loss_dict is not None:
        with open(os.path.join(m_path, "results", config['stage'], config['model_out'] + ".txt"), 'a') as file:
            for key, value in loss_dict.items():
                print(f'- {key}: {value}')
                file.write(f'- {key}: {value}\n')
                if 'wandb' in config and config['wandb']:
                    wandb.log({key: value})

    print('- Current RAM usage: %f GB'%(tracemalloc.get_traced_memory()[0]/1024/1024/1024))
    print('- Peak RAM usage: %f GB'%(tracemalloc.get_traced_memory()[1]/1024/1024/1024))
    if device.type == 'cuda':
        print("- Torch CUDA memory allocated: %f GB"%(torch.cuda.memory_allocated(torch.cuda.current_device())/1024/1024/1024))
        print("- Torch CUDA memory reserved: %f GB"%(torch.cuda.memory_reserved(torch.cuda.current_device())/1024/1024/1024))
        print("- Torch CUDA max memory reserved: %f GB"%(torch.cuda.max_memory_reserved(torch.cuda.current_device())/1024/1024/1024))
    with open(os.path.join(m_path, "results", config['stage'], config['model_out'] + ".txt"), 'a') as file:
        file.write('- Current RAM usage: %f GB\n'%(tracemalloc.get_traced_memory()[0]/1024/1024/1024))
        file.write('- Peak RAM usage: %f GB\n'%(tracemalloc.get_traced_memory()[1]/1024/1024/1024))
        if device.type == 'cuda':
            file.write("- Torch CUDA memory allocated: %f GB\n"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
            file.write("- Torch CUDA memory reserved: %f GB\n"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
            file.write("- Torch CUDA max memory reserved: %f GB\n"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))

    return

def plot_loss_graph(m_path, config, loss_list_dict):
    keys = list(loss_list_dict.keys())
    for idx, key in enumerate(keys):
        epoch_means = np.array(loss_list_dict[key])
        #epoch_means = np.mean(loss_values.reshape(-1, batch_number), axis=1)
        #epoch_stds = np.std(loss_values.reshape(-1, batch_number), axis=1)
        plt.figure(idx, figsize=(20, 20))
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.10f}'))
        plt.plot(range(len(epoch_means)), epoch_means, label="loss values", color="blue", linewidth=2.0)
        #plt.fill_between(range(len(epoch_stds)), epoch_means-epoch_stds, epoch_means+epoch_stds, color="blue", alpha=0.2)
        plt.axhline(y=epoch_means[-1], color="red", linestyle="dashed")
        plt.plot(len(epoch_means), epoch_means[-1], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="blue")
        plt.annotate("{:.3f}".format(epoch_means[-1]), xy=(len(epoch_means), epoch_means[-1]), horizontalalignment="left", verticalalignment="bottom")
        plt.xlabel("epoch")
        plt.ylabel(key)
        plt.title(f'{key} per epoch')
        plt.legend()
        plt.savefig(os.path.join(m_path, "results", config['stage'], config['model_out'] + f'_{key}.png'))
        plt.close(idx)

    return

def plot_metrics_bar(m_path, config, losses):
    keys = list(losses.keys())
    X_axis = np.arange(len(keys))
    lst_last = list(losses.values())
    last_losses = []
    for row in lst_last:
        if isinstance(row[-1], torch.Tensor):
            tensor = torch.unsqueeze(row[-1], dim=0)
            last_losses.extend(tensor.detach().cpu().numpy())
        else:
            last_losses.extend([row[-1]])
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
    metrics_bar = ax.bar(X_axis, last_losses, width=1, label="Loss values", align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.bar_label(metrics_bar)
    fig.legend()
    fig.savefig(os.path.join(m_path, "results", config['stage'], config['model_out'] + '_metrics.png'))
    plt.close()
    return

def save_trajectory(m_path, path, traj_feats):
    traj_arr = traj_feats.cpu().numpy()
    traj_arr = traj_arr.reshape((traj_arr.size))
    x_arr = traj_arr[0::2]
    y_arr = traj_arr[1::2]
    plt.figure(figsize=(20, 20))
    plt.plot(x_arr, y_arr)
    plt.savefig(os.path.join(m_path, path))
    plt.close()     
    return

def save_train_results(m_path, config, train_losses):
    plot_loss_graph(m_path, config, train_losses)
    plot_metrics_bar(m_path, config, train_losses)
    return

def save_test_results(m_path, config, loss_list_dict):
    plot_metrics_bar(m_path, config, loss_list_dict)
    return

def plot_loss_compare_graph(m_path, config, loss_dict, out_path):
    loss_list_dict = dict.fromkeys(loss_dict.keys(), [])
    model_in = config['model_outs'][0]
    for id in range(model_in, config['number_seeds'] + model_in):
        if "classifier" in config['stage']:
            res_path = os.path.join(m_path, "results", config['stage'], f"clf_{config['architecture']}_{config['dataset']}_exp{id}.txt")
        else:
            res_path = os.path.join(m_path, "results", config['stage'], f"{config['architecture']}_{config['dataset']}_exp{id}.txt")
    
        for loss_key in loss_list_dict.keys():
            with open(res_path, 'r') as res_file:
                tmp_loss = []
                for line in res_file:
                    if loss_key in line:
                        tmp_loss.append(np.double(line.removeprefix(f'- {loss_key}: ')))
            if len(tmp_loss) > 0:
                loss_list_dict[loss_key].append(tmp_loss)
    
    for idx, loss_key in enumerate(loss_dict.keys()):
        losses = np.array(list(map(list, zip(*loss_list_dict[loss_key]))))
        loss_means = np.mean(losses, axis=1)
        loss_stds = np.std(losses, axis=1)
        plt.figure(idx, figsize=(20, 20))
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.10f}'))
        plt.plot(range(len(loss_means)), loss_means, label="loss values", color="blue", linewidth=2.0)
        plt.fill_between(range(len(loss_stds)), loss_means-loss_stds, loss_means+loss_stds, color="blue", alpha=0.2)
        plt.axhline(y=loss_means[-1], color="red", linestyle="dashed")
        plt.plot(len(loss_means), loss_means[-1], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="blue")
        plt.annotate("{:.3f}".format(loss_means[-1]), xy=(len(loss_means), loss_means[-1]), horizontalalignment="left", verticalalignment="bottom")
        plt.xlabel("epoch")
        plt.ylabel(loss_key)
        plt.title(f'{loss_key} for {config["architecture"]} model')
        plt.legend()
        plt.savefig(os.path.join(m_path, "compare", config['stage'], f"{out_path}{loss_key}.png"))
        plt.close(idx)
    return

def plot_metric_compare_bar(m_path, config, loss_dict, out_path):
    if config["param_comp"] is not None:
        tmp_param_values = []

    for model_results in config['model_outs']:
        if "number_seeds" in config and config["number_seeds"] > 1:
            seed_results = [x for x in range(model_results, model_results + config['number_seeds'])]
        else:
            seed_results = [model_results]

        for id, seed_res in enumerate(seed_results):
            if "classifier" in config['stage']:
                seed_results[id] = os.path.join(m_path, "results", config['stage'], f"clf_{config['architecture']}_{config['dataset']}_exp{seed_res}.txt")
            else:
                seed_results[id] = os.path.join(m_path, "results", config['stage'], f"{config['architecture']}_{config['dataset']}_exp{seed_res}.txt")

        for loss_key in loss_dict.keys():
            tmp_loss = []
            for seed_res in seed_results:
                path = os.path.join(m_path, seed_res)
                with open(path, 'r') as res_file:
                    for line in res_file:
                        if config["param_comp"] is not None and config['param_comp'] in line:
                            tmp_param_values.append(np.double(line.removeprefix(f"{config['param_comp']}: ")))
                        if loss_key in line:
                            tmp_loss.append(np.double(line.removeprefix(f'- {loss_key}: ')))
            
            loss_dict[loss_key].append([np.mean(tmp_loss), np.std(tmp_loss)])
    
    if config["param_comp"] is not None:
        param_values = []
        [param_values.append(x) for x in tmp_param_values if x not in param_values]
        X_axis = np.arange(len(param_values))
        with open(os.path.join(m_path, "compare", config['stage'], out_path + 'metrics.txt'), 'a') as res_file:
            print(f'{config["param_comp"]} values: {param_values}')
            res_file.write(f'{config["param_comp"]} values: {param_values}\n')
    else:
        X_axis = np.arange(len(config['model_outs']))

    for loss_key in loss_dict.keys():
        loss_means, loss_stds = zip(*loss_dict[loss_key])
        if loss_key == "accuracy":
            lucky = list(zip(list(map(lambda x: round(x * 100, 2), loss_means)), list(map(lambda x: round(x * 100, 2), loss_stds))))
            for lu in lucky:
                print(f"{lu[0]} - {lu[1]}")
                print("#"*40)
            loss_means = tuple(map(lambda x: x * 100, loss_means))
            loss_stds = tuple(map(lambda x: x * 100, loss_stds))

        with open(os.path.join(m_path, "compare", config['stage'], out_path + 'metrics.txt'), 'a') as res_file:
            print(f"Values for {loss_key}:")
            print(f'- mean: {loss_means}')
            print(f'- std: {loss_stds}')
            res_file.write(f'Values for {loss_key}:\n')
            res_file.write(f'- mean: {loss_means}\n')
            res_file.write(f'- std: {loss_stds}\n')
        
        fig, ax = plt.subplots()
        fig.figsize=(20, 10)
        ax.set_xticks(X_axis)
        if config["param_comp"] is not None:
            ax.set_xticklabels(param_values)
            title = f"{loss_key} for diff. {config['param_comp']} values"
            if "parent_param" in config and config['parent_param'] is not None:
                title = title + f" for {config['parent_param']}"
            if "target_modality" in config and config['target_modality'] is not None:
                title = title + f" on the {config['target_modality']} mod."
        else:
            ax.set_xticklabels([config['architecture']])
            title = f"{loss_key} values"
            if "parent_param" in config and config['parent_param'] is not None:
                title = title + f" for {config['parent_param']}"
                if "target_modality" in config and config['target_modality'] is not None:
                    title = title + f" on the {config['target_modality']} mod."

        img_out_path = out_path + f"{loss_key}"
        ax.set_title(title)
        ax.yaxis.grid(True)
        metric_bar = ax.bar(X_axis, loss_means, yerr=loss_stds, width=0.4, align="center", alpha=0.5, ecolor='black', capsize=10)
        ax.bar_label(metric_bar)
        fig.savefig(os.path.join(m_path, "compare", config['stage'], img_out_path))
        plt.close()
    return

def plot_bar_across_models(m_path, config, out_path, architectures):
    loss_means = []
    loss_stds = []
    X_axis = np.arange(len(architectures))
    for architecture in architectures:
        path = os.path.join(m_path, "compare", config['stage'], f"{architecture}_{out_path}metrics.txt")
        with open(path, 'r') as res_file:
            file_lines = res_file.readlines()
            mean = file_lines[-2]
            std = file_lines[-1]
            mean = mean.removeprefix("- mean: (")
            mean = mean[:5]
            std = std.removeprefix("- std: (")
            std = std[:5]
            loss_means.append(np.double(mean))
            loss_stds.append(np.double(std))
    
    out_path = os.path.join(m_path, "compare", config['stage'], out_path + "accuracy.png")
    fig, ax = plt.subplots()
    fig.figsize=(25, 10)
    ax.set_xticks(X_axis)
    ax.set_xticklabels(architectures)
    ax.set_title(f"Accuracy in the {config['dataset']} dataset")
    ax.yaxis.grid(True)
    metric_bar = ax.bar(X_axis, loss_means, yerr=loss_stds, width=0.4, align="center", alpha=0.5, ecolor='black', capsize=10)
    ax.bar_label(metric_bar)
    plt.xticks(fontweight='light', fontsize='x-small')
    fig.savefig(os.path.join(m_path, "compare", config['stage'], out_path))
    plt.close()
    return