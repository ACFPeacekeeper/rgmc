import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import numpy as np
import tracemalloc
import argparse
import random
import torch
import time
import math
import sys
import os

from architectures import vae, dae
from input_transformations import gaussian_noise, fgsm
from matplotlib.ticker import StrMethodFormatter
from torchvision import transforms
from tqdm import tqdm
from collections import Counter

torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}.")

# Assign path to current directory
m_path = "copy-output-of-pwd-here"

def process_arguments():
    parser = argparse.ArgumentParser(prog="rmgm", description="Program tests the performance and robustness of several generative models with clean and noisy/adversarial samples.")
    parser.add_argument('model_type', choices=['VAE', 'DAE', 'GMC'], help='Model type to be used in the experiment.')
    parser.add_argument('-p', '--path_model', type=str, help="Filename of the file where the model is to be loaded from.")
    parser.add_argument('-m', '--model_out', type=str, help="Filename of the file where the model is to be saved to.")
    parser.add_argument('-d', '--dataset', type=str, default='MHD', choices=['MHD', 'MOSI_MOSEI', 'PENDULUM'], help='Dataset to be used in the experiments.')
    parser.add_argument('-s', '--stage', type=str, default='train_model', choices=['train_model', 'train_classifier', 'test_model', 'test_classifier'], help='Stage(s) of the pipeline to execute in the experiment.')
    parser.add_argument('-o', '--optimizer', type=str, default='adam', choices=['adam', 'none'], help='Define optimizer for the model.')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs to train the model.')
    parser.add_argument('-c', '--checkpoint', type=int, default=10, help='Epoch interval between checkpoints of the model in training.')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Number of samples processed for each model update.')
    parser.add_argument('-l', '--latent_dim', type=int, default=126, help='Dimension of the latent space of the models encodings.')
    parser.add_argument('-n', '--noise', type=str, default='none', choices=['none', 'gaussian'], help='Apply a type of noise to the model\'s input.')
    parser.add_argument('-a', '--adversarial_attack', '--attack', type=str, default='none', choices=['none', 'FGSM'], help='Execute an adversarial attack against the model.')
    parser.add_argument('-t', '--target_modality', type=str, default='image', choices=['image', 'trajectory'], help='Define the modality to target with noisy and/or adversarial samples.')
    parser.add_argument('--exclude_modality', type=str, default='none', choices=['none', 'image', 'trajectory'], help='Exclude a modality from the training/testing process.')
    parser.add_argument('--image_scale', type=float, default=1., help='Define weight for the image reconstruction loss.')
    parser.add_argument('--traj_scale', type=float, default=1., help='Define weight for the trajectory reconstruction loss.')
    parser.add_argument('--kld_beta', type=float, default=0.5, help='Define beta value for KL divergence.')
    parser.add_argument('--noise_mean', type=float, default=0., help='Define mean for noise distribution.')
    parser.add_argument('--noise_std', type=float, default=1., help='Define standard deviation for noise distribution.')
    parser.add_argument('--adv_eps', type=float, default=8/255, help='Define epsilon value for adversarial example generation.')
    parser.add_argument('--shuffle', type=bool, default=True, help='Shuffle the data after each epoch.')
    args = parser.parse_args()

    try:
        if args.stage == 'train_model':
            if args.epochs < 1:
                raise argparse.ArgumentError("Argument error: number of epochs must be a positive and non-zero integer.")
            elif args.batch_size < 1:
                raise argparse.ArgumentError("Argument error: batch_size value must be a positive and non-zero integer.")
            elif args.latent_dim < 1:
                raise argparse.ArgumentError("Argument error: latent_dim value must be a positive and non-zero integer.")
            elif args.checkpoint < 1:
                raise argparse.ArgumentError("Argument error: checkpoint value must be a positive and non-zero integer.")
            elif args.checkpoint > args.epochs:
                raise argparse.ArgumentError("Argument error: checkpoint value must be smaller than or equal to the number of epochs.")
    except argparse.ArgumentError:
        parser.print_help()
        sys.exit(1)

    os.makedirs(os.path.join(m_path, "results", args.stage), exist_ok=True)

    counter = 1
    file_path = os.path.join(m_path, "results", args.stage, f"{args.model_type.lower()}_{args.dataset.lower()}_{counter}.txt")
    while os.path.exists(file_path):
        counter += 1
        file_path = os.path.join(m_path, "results", args.stage, f"{args.model_type.lower()}_{args.dataset.lower()}_{counter}.txt")

    with open(file_path, 'w') as file:
        file.write(f'Model: {args.model_type}\n')
        print(f'Model: {args.model_type}')

        file.write(f'Exclude modality: {args.exclude_modality}\n')
        print(f'Exclude modality: {args.exclude_modality}')

        if args.exclude_modality == 'image':
            args.image_scale = 0.
            if args.target_modality == 'image':
                args.target_modality = 'none'
        elif args.exclude_modality == 'trajectory':
            args.traj_scale = 0.
            if args.target_modality == 'trajectory':
                args.target_modality = 'none'
        else:
            # If 2 modalities are present, divide scale of each modality by 2
            args.image_scale /= 2.
            args.traj_scale /= 2.

        if args.model_type == 'VAE' or args.model_type == 'DAE':
            file.write(f'Image recon loss scale: {args.image_scale}\n')
            print(f'Image recon loss scale: {args.image_scale}')
            file.write(f'Trajectory recon loss scale: {args.traj_scale}\n')
            print(f'Trajectory recon loss scale: {args.traj_scale}')
        if args.model_type == 'VAE':
            file.write(f'KLD beta value: {args.kld_beta}\n')
            print(f'KLD beta value: {args.kld_beta}')

        file.write(f'Dataset: {args.dataset}\n')
        print(f'Dataset: {args.dataset}')
        file.write(f'Latent dimension: {args.latent_dim}\n')
        print(f'Latent dimension: {args.latent_dim}')
        file.write(f'Pipeline stage: {args.stage}\n')
        print(f'Pipeline stage: {args.stage}')

    return args, file_path


def save_results(results_file_path, loss_dict):
    with open(results_file_path, 'a') as file:
        for key, value in loss_dict.items():
            print(f'{key}: {value}')
            file.write(f'- {key}: {value}\n')
    
    print('Current RAM usage: %f GB'%(tracemalloc.get_traced_memory()[0]/1024/1024/1024))
    print('Peak RAM usage: %f GB'%(tracemalloc.get_traced_memory()[1]/1024/1024/1024))
    if device.type == 'cuda':
        print("Torch CUDA memory allocated: %f GB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
        print("Torch CUDA memory reserved: %f GB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
        print("Torch CUDA max memory reserved: %f GB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
    with open(results_file_path, 'a') as file:
        file.write('- Current RAM usage: %f GB\n'%(tracemalloc.get_traced_memory()[0]/1024/1024/1024))
        file.write('- Peak RAM usage: %f GB\n'%(tracemalloc.get_traced_memory()[1]/1024/1024/1024))
        if device.type == 'cuda':
            file.write("- Torch CUDA memory allocated: %f GB\n"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
            file.write("- Torch CUDA memory reserved: %f GB\n"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
            file.write("- Torch CUDA max memory reserved: %f GB\n"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))

    return


def train_model(arguments, results_file_path):
    if arguments.model_type == 'VAE':
        model = vae.VAE(arguments.latent_dim, device, arguments.exclude_modality)
        loss_list_dict = {'Total loss': np.zeros(arguments.epochs), 'KLD': np.zeros(arguments.epochs), 'Img recon loss': np.zeros(arguments.epochs), 'Traj recon loss': np.zeros(arguments.epochs)}
        scales = {'Image recon scale': arguments.image_scale, 'Trajectory recon scale': arguments.traj_scale, 'KLD beta': arguments.kld_beta}
    elif arguments.model_type == 'DAE':
        model = dae.DAE(arguments.latent_dim, device, arguments.exclude_modality)
        loss_list_dict = {'Total loss': np.zeros(arguments.epochs), 'Img recon loss': np.zeros(arguments.epochs), 'Traj recon loss': np.zeros(arguments.epochs)}
        scales = {'Image recon scale': arguments.image_scale, 'Trajectory recon scale': arguments.traj_scale}

    model.cuda(device)

    print(f'Optimizer: {arguments.optimizer}')
    print(f'Batch size: {arguments.batch_size}')
    print(f'Shuffle data every epoch: {arguments.shuffle}')
    with open(results_file_path, 'a') as file:
        file.write(f'Optimizer: {arguments.optimizer}\n')
        file.write(f'Batch size: {arguments.batch_size}\n')
        file.write(f'Shuffle data every epoch: {arguments.shuffle}\n')
    
    if arguments.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters())

    checkpoint_counter = arguments.checkpoint 

    if arguments.dataset == 'MHD':
        dataset = torch.load(os.path.join(m_path, "datasets", "mhd", "dataset", "mhd_train.pt"))
    elif arguments.dataset == 'MOSI_MOSEI':
        raise NotImplementedError
    elif arguments.dataset == 'PENDULUM':
        dataset = torch.load(os.path.join(m_path, "datasets", "pendulum", "train_pendulum_dataset_samples20000_stack2_freq440.0_vel20.0_rec['LEFT_BOTTOM', 'RIGHT_BOTTOM', 'MIDDLE_TOP'].pt"))

    if arguments.exclude_modality == 'image':
        data = list(dataset[2].to(device))
    elif arguments.exclude_modality == 'trajectory':
        data = list(dataset[1].to(device))
    else:
        img_samples = dataset[1]
        traj_samples = dataset[2]
        img_samples = img_samples.to(device)
        traj_samples = traj_samples.to(device)
        data = list(zip(img_samples, traj_samples))

    print(f'Noise: {arguments.noise}')
    with open(results_file_path, 'a') as file:
        file.write(f'Noise: {arguments.noise}\n')
    
    if arguments.noise == "gaussian":
        noise = gaussian_noise.GaussianNoise(device, arguments.noise_mean, arguments.noise_std)
        data = noise.add_noise(data, arguments.target_modality)
        
        print(f'Noise mean: {arguments.noise_mean}')
        print(f'Noise standard deviation: {arguments.noise_std}')
        with open(results_file_path, 'a') as file:
            file.write(f'Noise mean: {arguments.noise_mean}\n')
            file.write(f'Noise standard deviation: {arguments.noise_std}\n')

    print(f'Adversarial attack: {arguments.adversarial_attack}')
    with open(results_file_path, 'a') as file:
        file.write(f'Adversarial attack: {arguments.adversarial_attack}\n')

    if arguments.adversarial_attack == 'FGSM':
        adv_attack = fgsm.FGSM(device, model, eps=arguments.adv_eps)
        data = adv_attack.example_generation(data, data, arguments.target_modality)

        print(f'FGSM epsilon value: {arguments.adv_eps}')
        with open(results_file_path, 'a') as file:
            file.write(f'FGSM epsilon value: {arguments.adv_eps}\n')

    if arguments.adversarial_attack != 'none' or arguments.noise != 'none':
        print(f'Target modality: {arguments.target_modality}')
        with open(results_file_path, 'a') as file:
            file.write(f'Target modality: {arguments.target_modality}\n')

    tracemalloc.start()
    for epoch in range(arguments.epochs):
        print(f'Epoch {epoch}')
        with open(results_file_path, 'a') as file:
            file.write(f'Epoch {epoch}:\n')

        loss_dict = Counter(dict.fromkeys(loss_list_dict.keys(), 0.))
            
        batch_number = math.ceil(len(data)/arguments.batch_size)

        if arguments.shuffle:
            random.shuffle(data)

        epoch_start = time.time()
        for batch_idx in tqdm(range(batch_number)):
            # Adjust batch size if its the last batch
            batch_end_idx = batch_idx*arguments.batch_size+arguments.batch_size if batch_idx*arguments.batch_size+arguments.batch_size < len(data) else len(data) 
            batch = data[batch_idx*arguments.batch_size:batch_end_idx]
            if arguments.optimizer != 'none':
                optimizer.zero_grad()
            
            x_hat, _ = model(batch)
            loss, batch_loss_dict = model.loss(batch, x_hat, scales=scales)
            loss_dict = loss_dict + batch_loss_dict
            
            loss.backward()
            if arguments.optimizer != 'none':
                optimizer.step()
        
        for key, value in loss_dict.items():
            loss_dict[key] = value / batch_number
            loss_list_dict[key][epoch] = loss_dict[key]
            
        epoch_end = time.time()
        print(f'Runtime: {epoch_end - epoch_start} sec')
        with open(results_file_path, 'a') as file:
            file.write(f'- Runtime: {epoch_end - epoch_start} sec\n')
        save_results(results_file_path, loss_dict)

        checkpoint_counter -= 1
        if checkpoint_counter == 0:
            print('Saving model checkpoint to file...')
            torch.save(model.state_dict(), os.path.join(m_path, "checkpoints", f'{arguments.model_type.lower()}_{arguments.dataset.lower()}_{epoch}.pt'))
            checkpoint_counter = arguments.checkpoint

        torch.cuda.empty_cache()
        tracemalloc.reset_peak()

    tracemalloc.stop()

    for idx, (key, values) in enumerate(loss_list_dict.items()):
        plt.figure(idx, figsize=(20, 20))
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.10f}'))
        plt.plot(range(arguments.epochs), values, label='loss values')
        plt.xlabel("Epoch")
        plt.ylabel(key)
        plt.title(f'{key} per epoch')
        plt.legend()
        plt.savefig(f'{os.path.splitext(results_file_path)[0]}_{key}.png')

    torch.save(model.state_dict(), os.path.join(m_path, "saved_models", arguments.model_out))
    torch.cuda.empty_cache()
    return model

def train_downstream_classifier(arguments, results_file_path):
    if arguments.model_type == 'VAE':
        model = vae.VAE(arguments.latent_dim, device)
        loss_list_dict = {'Total loss': np.zeros(arguments.epochs), 'KLD': np.zeros(arguments.epochs), 'Img recon loss': np.zeros(arguments.epochs), 'Traj recon loss': np.zeros(arguments.epochs)}
    elif arguments.model_type == 'DAE':
        model = dae.DAE(arguments.latent_dim, device)
        loss_list_dict = {'Total loss': np.zeros(arguments.epochs), 'Img recon loss': np.zeros(arguments.epochs), 'Traj recon loss': np.zeros(arguments.epochs)}

    if arguments.path_model:
        model.load_state_dict(torch.load(arguments.path_model))
        for param in model.parameters():
            param.requires_grad = False

    raise NotImplementedError
        


def test_model(arguments, results_file_path):
    if arguments.model_type == 'VAE':
        model = vae.VAE(arguments.latent_dim, device, arguments.exclude_modality)
        loss_dict = Counter({'Total loss': 0., 'KLD': 0., 'Img recon loss': 0., 'Traj recon loss': 0.})
        scales = {'Image recon scale': arguments.image_scale, 'Trajectory recon scale': arguments.traj_scale, 'KLD beta': arguments.kld_beta}
    elif arguments.model_type == 'DAE':
        model = dae.DAE(arguments.latent_dim, device, arguments.exclude_modality)
        loss_dict = Counter({'Total loss': 0., 'Img recon loss': 0., 'Traj recon loss': 0.})
        scales = {'Image recon scale': arguments.image_scale, 'Trajectory recon scale': arguments.traj_scale}

    if arguments.path_model:
        model.load_state_dict(torch.load(os.path.join("saved_models", arguments.path_model)))
        for param in model.parameters():
            param.requires_grad = False

    model.cuda(device)

    if arguments.dataset == 'MHD':
        dataset = torch.load(os.path.join(m_path, "datasets", "mhd", "dataset", "mhd_train.pt"))
    elif arguments.dataset == 'MOSI_MOSEI':
        raise NotImplementedError
    elif arguments.dataset == 'PENDULUM':
        dataset = torch.load(os.path.join(m_path, "datasets", "pendulum", "train_pendulum_dataset_samples20000_stack2_freq440.0_vel20.0_rec['LEFT_BOTTOM', 'RIGHT_BOTTOM', 'MIDDLE_TOP'].pt"))


    if arguments.exclude_modality == 'image':
        data = list(dataset[2].to(device))
    elif arguments.exclude_modality == 'trajectory':
        data = list(dataset[1].to(device))
    else:
        img_samples = dataset[1]
        traj_samples = dataset[2]
        img_samples = img_samples.to(device)
        traj_samples = traj_samples.to(device)
        data = list(zip(img_samples, traj_samples))

    print(f'Noise: {arguments.noise}')
    with open(results_file_path, 'a') as file:
        file.write(f'Noise: {arguments.noise}\n')
    
    if arguments.noise == "gaussian":
        noise = gaussian_noise.GaussianNoise(device, arguments.noise_mean, arguments.noise_std)
        data = noise.add_noise(data, arguments.target_modality)
        
        print(f'Noise mean: {arguments.noise_mean}')
        print(f'Noise standard deviation: {arguments.noise_std}')
        with open(results_file_path, 'a') as file:
            file.write(f'Noise mean: {arguments.noise_mean}\n')
            file.write(f'Noise standard deviation: {arguments.noise_std}\n')

    print(f'Adversarial attack: {arguments.adversarial_attack}')
    with open(results_file_path, 'a') as file:
        file.write(f'Adversarial attack: {arguments.adversarial_attack}\n')

    if arguments.adversarial_attack == 'FGSM':
        adv_attack = fgsm.FGSM(device, model, eps=arguments.adv_eps)
        data = adv_attack.example_generation(data, data, arguments.target_modality)

        print(f'FGSM epsilon value: {arguments.adv_eps}')
        with open(results_file_path, 'a') as file:
            file.write(f'FGSM epsilon value: {arguments.adv_eps}\n')

    if arguments.adversarial_attack != 'none' or arguments.noise != 'none':
        print(f'Target modality: {arguments.target_modality}')
        with open(results_file_path, 'a') as file:
            file.write(f'Target modality: {arguments.target_modality}\n')
    
    tracemalloc.start()
    print('Testing model')
    with open(results_file_path, 'a') as file:
        file.write('Testing model:\n')


    test_start = time.time()
    x_hat, _ = model(data)
    _, loss_dict = model.loss(data, x_hat, scales=scales)
    test_end = time.time()

    print(f'Runtime: {test_end - test_start} sec')
    with open(results_file_path, 'a') as file:
        file.write(f'- Runtime: {test_end - test_start} sec\n')
    save_results(results_file_path, loss_dict)
    tracemalloc.stop()
    return

def test_downstream_classifier(arguments):
    #TODO
    raise NotImplementedError


def main():
    os.makedirs(os.path.join(m_path, "results"), exist_ok=True)
    arguments, file_path = process_arguments()
    if arguments.stage == 'train_model':
        os.makedirs(os.path.join(m_path, "saved_models"), exist_ok=True)
        os.makedirs(os.path.join(m_path, "checkpoints"), exist_ok=True)
        train_model(arguments, file_path)
    elif arguments.stage == 'train_classifier':
        os.makedirs(os.path.join(m_path, "saved_models"), exist_ok=True)
        os.makedirs(os.path.join(m_path, "checkpoints"), exist_ok=True)
        train_downstream_classifier(arguments, file_path)
    elif arguments.stage == 'test_model':
        test_model(arguments, file_path)
    elif arguments.stage == 'test_classifier':
        test_downstream_classifier(arguments, file_path)

if __name__ == "__main__":
    main()