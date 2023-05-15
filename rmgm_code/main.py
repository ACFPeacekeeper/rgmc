import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import numpy as np
import tracemalloc
import traceback
import argparse
import random
import torch
import time
import math
import sys
import os

from architectures import vae, dae, classifier
from input_transformations import gaussian_noise, fgsm
from matplotlib.ticker import StrMethodFormatter
from torchvision import transforms
from tqdm import tqdm
from collections import Counter

torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}.")

# Assign path to current directory
m_path = "/home/pkhunter/Repositories/rmgm/rmgm_code"

def process_arguments():
    parser = argparse.ArgumentParser(prog="rmgm", description="Program tests the performance and robustness of several generative models with clean and noisy/adversarial samples.")
    parser.add_argument('model_type', choices=['VAE', 'DAE', 'GMC'], help='Model type to be used in the experiment.')
    parser.add_argument('-p', '--path_model', type=str, default='none', help="Filename of the file where the model is to be loaded from.")
    parser.add_argument('--path_classifier', type=str, default='none', help="Filename of the file where the classifier is to be loaded from.")
    parser.add_argument('-m', '--model_out', type=str, default='none', help="Filename of the file where the model is to be saved to.")
    parser.add_argument('-d', '--dataset', type=str, default='MHD', choices=['MHD', 'MOSI_MOSEI', 'PENDULUM'], help='Dataset to be used in the experiments.')
    parser.add_argument('-s', '--stage', type=str, default='train_model', choices=['train_model', 'train_classifier', 'test_model', 'test_classifier', 'inference'], help='Stage of the pipeline to execute in the experiment.')
    parser.add_argument('-o', '--optimizer', type=str, default='SGD', choices=['adam', 'SGD', 'none'], help='Optimizer for the model training process.')
    parser.add_argument('-r', '--learning_rate', '--lr', type=float, default=0.01, help='Learning rate value for the optimizer.')
    parser.add_argument('-e', '--epochs', type=int, default=50, help='Number of epochs to train the model.')
    parser.add_argument('-c', '--checkpoint', type=int, default=50, help='Epoch interval between checkpoints of the model in training.')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Number of samples processed for each model update.')
    parser.add_argument('-l', '--latent_dim', type=int, default=128, help='Dimension of the latent space of the models encodings.')
    parser.add_argument('-n', '--noise', type=str, default='none', choices=['none', 'gaussian'], help='Apply a type of noise to the model\'s input.')
    parser.add_argument('-a', '--adversarial_attack', '--attack', type=str, default='none', choices=['none', 'FGSM'], help='Execute an adversarial attack against the model.')
    parser.add_argument('-t', '--target_modality', type=str, default='image', choices=['image', 'trajectory'], help='Modality to target with noisy and/or adversarial samples.')
    parser.add_argument('--exclude_modality', type=str, default='none', choices=['none', 'image', 'trajectory'], help='Exclude a modality from the training/testing process.')
    parser.add_argument('--image_scale', type=float, default=1., help='Weight for the image reconstruction loss.')
    parser.add_argument('--traj_scale', type=float, default=1., help='Weight for the trajectory reconstruction loss.')
    parser.add_argument('--kld_betas', nargs=2, type=float, default=[0., 1.], help='Min and max beta values for KL divergence.')
    parser.add_argument('--vae_mean', type=float, default=0., help='Mean value for the reparameterization trick for the VAE.')
    parser.add_argument('--vae_std', type=float, default=1., help='Standard deviation value for the reparameterization trick for the VAE.')
    parser.add_argument('--adam_betas', nargs=2, type=float, default=[0.9, 0.999], help='Beta values for the Adam optimizer.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for the SGD optimizer.')
    parser.add_argument('--noise_mean', type=float, default=0., help='Mean for noise distribution.')
    parser.add_argument('--noise_std', type=float, default=1., help='Standard deviation for noise distribution.')
    parser.add_argument('--adv_eps', type=float, default=8/255, help='Epsilon value for adversarial example generation.')
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
        else:
            if args.path_model == 'none':
                raise argparse.ArgumentError(f"Argument error: the --path_model argument cannot be none when the --stage argument is {args.stage}.")
        if args.exclude_modality == args.target_modality:
            raise argparse.ArgumentError("Argument error: target modality cannot be the same as excluded modality.")
    
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

        if args.model_type == 'VAE':
            file.write(f'Reparameterization trick mean: {args.vae_mean}\n')
            print(f'Reparameterization trick mean: {args.vae_mean}')
            file.write(f'Reparameterization trick standard deviation: {args.vae_std}\n')
            print(f'Reparameterization trick standard deviation: {args.vae_std}')

        file.write(f'Exclude modality: {args.exclude_modality}\n')
        print(f'Exclude modality: {args.exclude_modality}')

        if args.path_model != 'none':
            file.write(f'Load model file: {args.path_model}\n')
            print(f'Load model file: {args.path_model}')
            if args.stage == 'test_classifier':
                if args.path_classifier == 'none':
                    args.path_classifier = os.path.join(os.path.dirname(args.path_model), "clf_" + os.path.basename(args.path_model))
                file.write(f'Load classifier file: {args.path_classifier}\n')
                print(f'Load classifier file: {args.path_classifier}')
            
        elif args.stage == 'inference':
            file.write(f'Load model file: {os.path.basename(args.path_model)}\n')
            print(f'Load model file: {os.path.basename(args.path_model)}')

        if args.model_out != 'none':
            if args.stage == 'train_model':
                file.write(f'Store model file: saved_models/{args.model_out}\n')
                print(f'Store model file: saved_models/{args.model_out}')
            elif args.stage == 'train_classifier':
                file.write(f'Store classifier file: saved_models/{args.model_out}\n')
                print(f'Store classifier file: saved_models/{args.model_out}')
        elif args.stage == 'train_classifier':
            file.write(f'Store classifier file: saved_models/clf_{os.path.basename(args.path_model)}\n')
            print(f'Store classifier file: saved_models/clf_{os.path.basename(args.path_model)}')


        if args.stage == 'train_model' or args.stage == 'train_classifier' or args.stage == 'inference':
            file.write(f'Checkpoint save counter: {args.checkpoint}\n')
            print(f'Checkpoint save counter: {args.checkpoint}')
        

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
            if args.stage == 'train_model':
                file.write(f'KLD betas value: {args.kld_betas}\n')
                print(f'KLD betas value: {args.kld_betas}')
            else:
                file.write(f'KLD beta value: {args.kld_betas[1]}\n')
                print(f'KLD beta value: {args.kld_betas[1]}')

        file.write(f'Dataset: {args.dataset}\n')
        print(f'Dataset: {args.dataset}')
        file.write(f'Latent dimension: {args.latent_dim}\n')
        print(f'Latent dimension: {args.latent_dim}')
        file.write(f'Pipeline stage: {args.stage}\n')
        print(f'Pipeline stage: {args.stage}')

    return args, file_path

def dataset_setup(arguments, results_file_path, model, get_labels=False):
    if arguments.dataset == 'MHD':
        dataloader = torch.load(os.path.join(m_path, "datasets", "mhd", "dataset", "mhd_train.pt"))
    elif arguments.dataset == 'MOSI_MOSEI':
        raise NotImplementedError
    elif arguments.dataset == 'PENDULUM':
        dataloader = torch.load(os.path.join(m_path, "datasets", "pendulum", "train_pendulum_dataset_samples20000_stack2_freq440.0_vel20.0_rec['LEFT_BOTTOM', 'RIGHT_BOTTOM', 'MIDDLE_TOP'].pt"))

    if arguments.exclude_modality == 'image':
        dataset = {'trajectory': dataloader[2].to(device)}
    elif arguments.exclude_modality == 'trajectory':
        dataset = {'image': dataloader[1].to(device)}
    else:
        dataset = {'image': dataloader[1].to(device), 'trajectory': dataloader[2].to(device)}

    if get_labels:
        dataset['label'] = dataloader[0].to(device)

    print(f'Noise: {arguments.noise}')
    with open(results_file_path, 'a') as file:
        file.write(f'Noise: {arguments.noise}\n')
    
    if arguments.noise == "gaussian":
        noise = gaussian_noise.GaussianNoise(device, arguments.noise_mean, arguments.noise_std)
        dataset[arguments.target_modality] = noise.add_noise(dataset[arguments.target_modality])

        print(f'Noise mean: {arguments.noise_mean}')
        print(f'Noise standard deviation: {arguments.noise_std}')
        with open(results_file_path, 'a') as file:
            file.write(f'Noise mean: {arguments.noise_mean}\n')
            file.write(f'Noise standard deviation: {arguments.noise_std}\n')

    print(f'Adversarial attack: {arguments.adversarial_attack}')
    with open(results_file_path, 'a') as file:
        file.write(f'Adversarial attack: {arguments.adversarial_attack}\n')

    if arguments.adversarial_attack == 'FGSM':
        adv_attack = fgsm.FGSM(device, model, arguments.target_modality, eps=arguments.adv_eps,)
        dataset[arguments.target_modality] = adv_attack.example_generation(dataset, dataset[arguments.target_modality])

        print(f'FGSM epsilon value: {arguments.adv_eps}')
        with open(results_file_path, 'a') as file:
            file.write(f'FGSM epsilon value: {arguments.adv_eps}\n')

    if arguments.adversarial_attack != 'none' or arguments.noise != 'none':
        print(f'Target modality: {arguments.target_modality}')
        with open(results_file_path, 'a') as file:
            file.write(f'Target modality: {arguments.target_modality}\n')

    return dataset


def save_results(results_file_path, loss_dict=None):
    if loss_dict is not None:
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

def save_final_results(arguments, results_file_path, loss_list_dict):
    for idx, (key, values) in enumerate(loss_list_dict.items()):
        plt.figure(idx, figsize=(20, 20))
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.10f}'))
        plt.plot(range(arguments.epochs), values, label='loss values')
        plt.xlabel("Epoch")
        plt.ylabel(key)
        plt.title(f'{key} per epoch')
        plt.legend()
        plt.savefig(f'{os.path.splitext(results_file_path)[0]}_{key}.png')

    with open(results_file_path, 'a') as file:
        print('Average epoch results:')
        file.write('Average epoch results:\n')
        for key, values in loss_list_dict.items():
            print(f'- {key}: {np.mean(values)}')
            file.write(f'- {key}: {np.mean(values)}\n')
            

    return


def train_model(arguments, results_file_path):
    if arguments.model_type == 'VAE':
        scales = {'image': arguments.image_scale, 'trajectory': arguments.traj_scale, 'KLD betas': arguments.kld_betas}
        model = vae.VAE(arguments.model_type, arguments.latent_dim, device, arguments.exclude_modality, scales, arguments.vae_mean, arguments.vae_std)
        loss_list_dict = {'Total loss': np.zeros(arguments.epochs), 'KLD': np.zeros(arguments.epochs), 'Img recon loss': np.zeros(arguments.epochs), 'Traj recon loss': np.zeros(arguments.epochs)}
    elif arguments.model_type == 'DAE':
        scales = {'image': arguments.image_scale, 'trajectory': arguments.traj_scale}
        model = dae.DAE(arguments.model_type, arguments.latent_dim, device, arguments.exclude_modality, scales)
        loss_list_dict = {'Total loss': np.zeros(arguments.epochs), 'Img recon loss': np.zeros(arguments.epochs), 'Traj recon loss': np.zeros(arguments.epochs)}

    model.cuda(device)

    print(f'Optimizer: {arguments.optimizer}')
    print(f'Batch size: {arguments.batch_size}')
    with open(results_file_path, 'a') as file:
        file.write(f'Optimizer: {arguments.optimizer}\n')
        file.write(f'Batch size: {arguments.batch_size}\n')

    if arguments.optimizer != 'none':
        print(f'Learning rate: {arguments.learning_rate}')
        with open(results_file_path, 'a') as file:
            file.write(f'Learning rate: {arguments.learning_rate}\n')

        if arguments.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=arguments.learning_rate, betas=arguments.adam_betas)
            print(f'Adam betas: {arguments.adam_betas}')
            with open(results_file_path, 'a') as file:
                file.write(f'Adam betas: {arguments.adam_betas}\n')
        elif arguments.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=arguments.learning_rate)
            print(f'SGD momentum: {arguments.momentum}')
            with open(results_file_path, 'a') as file:
                file.write(f'SGD momentum: {arguments.momentum}\n')

    checkpoint_counter = arguments.checkpoint 

    dataset = dataset_setup(arguments, results_file_path, model)

    tracemalloc.start()
    for epoch in range(arguments.epochs):
        print(f'Epoch {epoch}')
        with open(results_file_path, 'a') as file:
            file.write(f'Epoch {epoch}:\n')

        loss_dict = Counter(dict.fromkeys(loss_list_dict.keys(), 0.))
            
        batch_number = math.ceil(len(list(dataset.values())[0])/arguments.batch_size)

        epoch_start = time.time()
        for batch_idx in tqdm(range(batch_number)):
            # Adjust batch size if its the last batch
            batch_end_idx = batch_idx*arguments.batch_size+arguments.batch_size if batch_idx*arguments.batch_size+arguments.batch_size < len(list(dataset.values())[0]) else len(list(dataset.values())[0])
            batch = dict.fromkeys(dataset.keys())
            for key, value in dataset.items():
                batch[key] = value[batch_idx*arguments.batch_size:batch_end_idx, :]

            if arguments.optimizer != 'none':
                optimizer.zero_grad()
        
            x_hat, _ = model(batch)    
            loss, batch_loss_dict = model.loss(batch, x_hat)
            loss_dict = loss_dict + batch_loss_dict
            
            loss.backward()
            if arguments.optimizer != 'none':
                optimizer.step()

            if model.name == 'VAE':
                kld_weight = 2 * batch_idx / batch_number
                model.update_kld_scale(kld_weight)
        
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
    save_final_results(arguments, results_file_path, loss_list_dict)
    if arguments.model_out != 'none':
        torch.save(model.state_dict(), os.path.join(m_path, "saved_models", arguments.model_out))
    else:
        torch.save(model.state_dict(), os.path.join(m_path, "saved_models", f'{os.path.basename(os.path.splitext(results_file_path)[0])}.pt'))
    torch.cuda.empty_cache()
    return model


def train_downstream_classifier(arguments, results_file_path):
    if arguments.model_type == 'VAE':
        scales = {'image': arguments.image_scale, 'trajectory': arguments.traj_scale, 'KLD betas': arguments.kld_betas}
        model = vae.VAE(arguments.model_type, arguments.latent_dim, device, arguments.exclude_modality, scales, arguments.vae_mean, arguments.vae_std, test=True)
    elif arguments.model_type == 'DAE':
        scales = {'image': arguments.image_scale, 'trajectory': arguments.traj_scale}
        model = dae.DAE(arguments.model_type, arguments.latent_dim, device, arguments.exclude_modality, scales, test=True)

    loss_list_dict = {'Loss': np.zeros(arguments.epochs)}

    model.load_state_dict(torch.load(arguments.path_model))
    for param in model.parameters():
        param.requires_grad = False

    model.cuda(device)

    clf = classifier.MNISTClassifier(arguments.latent_dim, model)

    clf.cuda(device)

    print(f'Optimizer: {arguments.optimizer}')
    print(f'Batch size: {arguments.batch_size}')
    with open(results_file_path, 'a') as file:
        file.write(f'Optimizer: {arguments.optimizer}\n')
        file.write(f'Batch size: {arguments.batch_size}\n')
    
    if arguments.optimizer != 'none':
        print(f'Learning rate: {arguments.learning_rate}')
        with open(results_file_path, 'a') as file:
            file.write(f'Learning rate: {arguments.learning_rate}\n')

        if arguments.optimizer == 'adam':
            optimizer = optim.Adam(clf.parameters(), lr=arguments.learning_rate, betas=arguments.adam_betas)
            print(f'Adam betas: {arguments.adam_betas}')
            with open(results_file_path, 'a') as file:
                file.write(f'Adam betas: {arguments.adam_betas}\n')
        elif arguments.optimizer == 'SGD':
            optimizer = optim.SGD(clf.parameters(), lr=arguments.learning_rate)
            print(f'SGD momentum: {arguments.momentum}')
            with open(results_file_path, 'a') as file:
                file.write(f'SGD momentum: {arguments.momentum}\n')

    checkpoint_counter = arguments.checkpoint 

    dataset = dataset_setup(arguments, results_file_path, model, get_labels=True)

    tracemalloc.start()
    for epoch in range(arguments.epochs):
        print(f'Epoch {epoch}')
        with open(results_file_path, 'a') as file:
            file.write(f'Epoch {epoch}:\n')
            
        batch_number = math.ceil(len(list(dataset.values())[0])/arguments.batch_size)

        if arguments.optimizer != 'none':
                optimizer.zero_grad()

        loss_dict = dict.fromkeys(loss_list_dict.keys(), 0.)

        epoch_start = time.time()

        for batch_idx in tqdm(range(batch_number)):
            # Adjust batch size if its the last batch
            batch_end_idx = batch_idx*arguments.batch_size+arguments.batch_size if batch_idx*arguments.batch_size+arguments.batch_size < len(list(dataset.values())[0]) else len(list(dataset.values())[0])
            batch = dict.fromkeys(dataset.keys())
            batch.pop('label', None)
            for key, value in dataset.items():
                if key != 'label':
                    batch[key] = value[batch_idx*arguments.batch_size:batch_end_idx, :]
                else:
                    batch_labels = value[batch_idx*arguments.batch_size:batch_end_idx]

            if arguments.optimizer != 'none':
                optimizer.zero_grad()

            _, classification, _ = clf(batch)
            loss = clf.loss(classification, batch_labels)

            loss.backward()
            if arguments.optimizer != 'none':
                optimizer.step()

            loss_dict['Loss'] += loss

        loss_dict['Loss'] = loss_dict['Loss'] / batch_number
        loss_list_dict['Loss'][epoch] = loss_dict['Loss']

        epoch_end = time.time()
        print(f'Runtime: {epoch_end - epoch_start} sec')
        with open(results_file_path, 'a') as file:
            file.write(f'- Runtime: {epoch_end - epoch_start} sec\n')
        save_results(results_file_path, loss_dict)

        checkpoint_counter -= 1
        if checkpoint_counter == 0:
            print('Saving model checkpoint to file...')
            torch.save(clf.state_dict(), os.path.join(m_path, "checkpoints", f'clf_{arguments.model_type.lower()}_{arguments.dataset.lower()}_{epoch}.pt'))
            checkpoint_counter = arguments.checkpoint

        torch.cuda.empty_cache()
        tracemalloc.reset_peak()

    tracemalloc.stop()
    save_final_results(arguments, results_file_path, loss_list_dict)
    if arguments.model_out != 'none':
        torch.save(clf.state_dict(), os.path.join(m_path, "saved_models", arguments.model_out))
    else:
        torch.save(clf.state_dict(), os.path.join(m_path, "saved_models", f'clf_{os.path.basename(os.path.splitext(results_file_path)[0])}.pt'))
    torch.cuda.empty_cache()
    return clf


def test_model(arguments, results_file_path):
    if arguments.model_type == 'VAE':
        scales = {'image': arguments.image_scale, 'trajectory': arguments.traj_scale, 'KLD betas': arguments.kld_betas}
        model = vae.VAE(arguments.model_type, arguments.latent_dim, device, arguments.exclude_modality, scales, arguments.vae_mean, arguments.vae_std, test=True)
        loss_dict = {'Total loss': 0., 'KLD': 0., 'Img recon loss': 0., 'Traj recon loss': 0.}
    elif arguments.model_type == 'DAE':
        scales = {'image': arguments.image_scale, 'trajectory': arguments.traj_scale}
        model = dae.DAE(arguments.model_type, arguments.latent_dim, device, arguments.exclude_modality, scales, test=True)
        loss_dict = {'Total loss': 0., 'Img recon loss': 0., 'Traj recon loss': 0.}

    model.load_state_dict(torch.load(arguments.path_model))
    for param in model.parameters():
        param.requires_grad = False

    model.cuda(device)

    dataset = dataset_setup(arguments, results_file_path, model)
    
    tracemalloc.start()
    print('Testing model')
    with open(results_file_path, 'a') as file:
        file.write('Testing model:\n')


    test_start = time.time()
    x_hat, _ = model(dataset)
    _, loss_dict = model.loss(dataset, x_hat)
    test_end = time.time()

    print(f'Runtime: {test_end - test_start} sec')
    with open(results_file_path, 'a') as file:
        file.write(f'- Runtime: {test_end - test_start} sec\n')
    save_results(results_file_path, loss_dict)
    tracemalloc.stop()
    return

def test_downstream_classifier(arguments, results_file_path):
    if arguments.model_type == 'VAE':
        scales = {'image': arguments.image_scale, 'trajectory': arguments.traj_scale, 'KLD betas': arguments.kld_betas}
        model = vae.VAE(arguments.model_type, arguments.latent_dim, device, arguments.exclude_modality, scales, arguments.vae_mean, arguments.vae_std, test=True)
    elif arguments.model_type == 'DAE':
        scales = {'image': arguments.image_scale, 'trajectory': arguments.traj_scale}
        model = dae.DAE(arguments.model_type, arguments.latent_dim, device, arguments.exclude_modality, scales, test=True)

    model.load_state_dict(torch.load(arguments.path_model))
    for param in model.parameters():
        param.requires_grad = False

    model.cuda(device)

    clf = classifier.MNISTClassifier(arguments.latent_dim, model)
    clf_path = os.path.join("saved_models", "clf_" + os.path.basename(arguments.path_model))
    clf.load_state_dict(torch.load(clf_path))
    for param in clf.parameters():
        param.requires_grad = False

    clf.cuda(device)

    dataset = dataset_setup(arguments, results_file_path, model, get_labels=True)

    tracemalloc.start()
    print('Testing classifier')
    with open(results_file_path, 'a') as file:
        file.write('Testing classifier:\n')

    features =  dataset.copy()
    features.pop('label', None)
    labels = dataset['label']

    test_start = time.time()
    x_hat, classification, _ = clf(features)
    _, loss_dict = clf.model.loss(features, x_hat)
    clf_loss = clf.loss(classification, labels)
    test_end = time.time()


    loss_dict['Classifier loss'] = clf_loss
    print(f'Runtime: {test_end - test_start} sec')
    with open(results_file_path, 'a') as file:
        file.write(f'- Runtime: {test_end - test_start} sec\n')
    save_results(results_file_path, loss_dict)
    tracemalloc.stop()
    return

def inference(arguments, results_file_path):
    if arguments.model_type == 'VAE':
        scales = {'image': arguments.image_scale, 'trajectory': arguments.traj_scale, 'KLD betas': arguments.kld_betas}
        model = vae.VAE(arguments.latent_dim, device, arguments.exclude_modality, scales, arguments.vae_mean, arguments.vae_std)
    elif arguments.model_type == 'DAE':
        scales = {'image': arguments.image_scale, 'trajectory': arguments.traj_scale}
        model = dae.DAE(arguments.latent_dim, device, arguments.exclude_modality, scales, test=True)

    model.load_state_dict(torch.load(arguments.path_model))
    for param in model.parameters():
        param.requires_grad = False

    model.cuda(device)

    dataset = dataset_setup(arguments, results_file_path, model)
    
    tracemalloc.start()
    print('Performing inference')
    with open(results_file_path, 'a') as file:
        file.write('Performing inference:\n')


    inference_start = time.time()
    x_hat, _ = model(dataset)
    counter = 0
    for idx, (img, recon) in tqdm(enumerate(zip(dataset['image'], x_hat['image'])), total=x_hat['image'].size(dim=0)):
        if counter % arguments.checkpoint == 0: 
            img_path = os.path.basename(os.path.splitext(results_file_path)[0])
            plt.imsave(os.path.join("images", f'{img_path}_{idx}_orig.png'), torch.reshape(img, (28,28)).detach().clone().cpu())
            plt.imsave(os.path.join("images", f'{img_path}_{idx}_recon.png'), torch.reshape(recon, (28,28)).detach().clone().cpu())
        counter += 1

    inference_stop = time.time()
    print(f'Runtime: {inference_stop - inference_start} sec')
    with open(results_file_path, 'a') as file:
        file.write(f'- Runtime: {inference_stop - inference_start} sec\n')
    save_results(results_file_path)
    tracemalloc.stop()
    return


def main():
    os.makedirs(os.path.join(m_path, "results"), exist_ok=True)
    arguments, file_path = process_arguments()
    try:
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
        elif arguments.stage == 'inference':
            os.makedirs(os.path.join(m_path, "images"), exist_ok=True)
            inference(arguments, file_path)

    except:
        traceback.print_exception(*sys.exc_info())
        os.remove(file_path)
        sys.exit(1)

if __name__ == "__main__":
    main()
    sys.exit(0)