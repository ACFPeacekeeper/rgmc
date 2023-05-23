import re
import os
import sys
import json
import math
import torch
import wandb
import select
import shutil
import pickle
import termios
import argparse
import itertools
import traceback
import subprocess

import numpy as np
import torch.optim as optim

from datasets.mhd.MHDDataset import MHDDataset
from datasets.mosi.MOSIDataset import MOSIDataset
from datasets.mosei.MOSEIDataset import MOSEIDataset
from datasets.pendulum.PendulumDataset import PendulumDataset
from input_transformations import gaussian_noise, fgsm
from architectures import vae, dae, gmc, mvae, classifier

TIMEOUT = 0 # Seconds to wait for user to input notes
ARCHITECTURES = ['vae', 'dae', 'gmc', 'mvae']
DATASETS = ['mhd', 'mosi', 'mosei', 'pendulum']
OPTIMIZERS = ['sgd', 'adam', None]
NOISE_TYPES = ['gaussian', None] 
ADVERSARIAL_ATTACKS = ["fgsm", None]
EXPERTS_FUSION_TYPES = ['poe', 'moe', None]
MODALITIES = ['image', 'trajectory', None]
STAGES = ['train_model', 'train_classifier', 'test_model', 'test_classifier', 'inference']

SEED = 42
LR_DEFAULT = 0.001
EPOCHS_DEFAULT = 10
BATCH_SIZE_DEFAULT = 256
CHECKPOINT_DEFAULT = 0
LATENT_DIM_DEFAULT = 128
INFONCE_TEMPERATURE_DEFAULT = 0.2
RECON_SCALE_DEFAULTS = {'image': 0.5, 'trajectory': 0.5}
KLD_BETA_DEFAULT = 0.5
REPARAMETERIZATION_MEAN_DEFAULT = 0.0
REPARAMETERIZATION_STD_DEFAULT = 1.0
POE_EPS_DEFAULT = 1e-8
MOMENTUM_DEFAULT = 0.9
ADAM_BETAS_DEFAULTS = [0.9, 0.999]
NOISE_MEAN_DEFAULT = 0.0
NOISE_STD_DEFAULT = 1.0
ADV_EPSILON_DEFAULT = 8 / 255

def process_arguments(m_path):
    parser = argparse.ArgumentParser(prog="rmgm", description="Program tests the performance and robustness of several generative models with clean and noisy/adversarial samples.")
    subparsers = parser.add_subparsers(help="command", dest="command")
    clear_parser = subparsers.add_parser("clear")
    clear_parser.add_argument('--clear_results', '--clear_res', action="store_false", help="Flag to delete results directory.")
    clear_parser.add_argument('--clear_checkpoints', '--clear_check', action="store_false", help="Flag to delete checkpoints directory.")
    clear_parser.add_argument('--clear_saved_models', '--clear_models', '--clear_saved', action="store_false", help="Flag to delete saved_models directory.")
    clear_parser.add_argument('--clear_wandb', '--clear_w&b', action="store_false", help="Flag to delete wandb directory.")
    clear_parser.add_argument('--clear_configs', '--clear_runs', action="store_false", help="Flag to delete configs directory.")
    clear_parser.add_argument('--clear_idx', action='store_false', help="Flag to delete previous experiments idx file.")

    configs_parser = subparsers.add_parser("config")
    configs_parser.add_argument('--load_config', '--load_json', type=str, help='File path where the experiment(s) configurations are to loaded from.')
    configs_parser.add_argument('--config_permute', '--config_permutations', type=str, help='Generate several config runs from permutations of dict of lists with hyperparams.')
    configs_parser.add_argument('--seed', '--torch_seed', type=int, default=SEED, help='Seed value for results replication.')

    exp_parser = subparsers.add_parser("experiment")
    exp_parser.add_argument('--config_out', '--json_out', type=str, default=None, help='File path where the experiment configurations are to saved to.')
    exp_parser.add_argument('-a', '--architecture', choices=ARCHITECTURES, help='Architecture to be used in the experiment.')
    exp_parser.add_argument('-p', '--path_model', type=str, default=None, help="Filename of the file where the model is to be loaded from.")
    exp_parser.add_argument('--seed', '--torch_seed', '--pytorch_seed', type=int, default=SEED, help='Seed value for results replication.')
    exp_parser.add_argument('--load_config', '--load_json', type=str, default=None, help='Filename of config file of the model training, to load model config from.')
    exp_parser.add_argument('--path_classifier', type=str, default=None, help="Filename of the file where the classifier is to be loaded from.")
    exp_parser.add_argument('-m', '--model_out', type=str, default=None, help="Filename of the file where the model/classifier is to be saved to.")
    exp_parser.add_argument('-d', '--dataset', type=str, default='mhd', choices=DATASETS, help='Dataset to be used in the experiments.')
    exp_parser.add_argument('-s', '--stage', type=str, default='train_model', choices=STAGES, help='Stage of the pipeline to execute in the experiment.')
    exp_parser.add_argument('-o', '--optimizer', type=str, default='sgd', choices=OPTIMIZERS, help='Optimizer for the model training process.')
    exp_parser.add_argument('-r', '--learning_rate', '--lr', type=float, default=LR_DEFAULT, help='Learning rate value for the optimizer.')
    exp_parser.add_argument('-e', '--epochs', type=int, default=EPOCHS_DEFAULT, help='Number of epochs to train the model.')
    exp_parser.add_argument('-b', '--batch_size', type=int, default=BATCH_SIZE_DEFAULT, help='Number of samples processed for each model update.')
    exp_parser.add_argument('--checkpoint', type=int, default=CHECKPOINT_DEFAULT, help='Epoch interval between checkpoints of the model in training.')
    exp_parser.add_argument('--latent_dimension', '--latent_dimension', type=int, default=LATENT_DIM_DEFAULT, help='Dimension of the latent space of the models encodings.')
    exp_parser.add_argument('--noise', type=str, default=None, choices=NOISE_TYPES, help='Apply a type of noise to the model\'s input.')
    exp_parser.add_argument('--adversarial_attack', '--attack', type=str, default=None, choices=ADVERSARIAL_ATTACKS, help='Execute an adversarial attack against the model.')
    exp_parser.add_argument('--target_modality', type=str, default=None, choices=MODALITIES, help='Modality to target with noisy and/or adversarial samples.')
    exp_parser.add_argument('--exclude_modality', type=str, default=None, choices=MODALITIES, help='Exclude a modality from the training/testing process.')
    exp_parser.add_argument('--infonce_temperature', '--infonce_temp', type=float, default=INFONCE_TEMPERATURE_DEFAULT, help='Temperature for the infonce loss.')
    exp_parser.add_argument('--image_recon_scale', type=float, default=RECON_SCALE_DEFAULTS['image'], help='Weight for the image reconstruction loss.')
    exp_parser.add_argument('--traj_recon_scale', type=float, default=RECON_SCALE_DEFAULTS['trajectory'], help='Weight for the trajectory reconstruction loss.')
    exp_parser.add_argument('--kld_beta', type=float, default=KLD_BETA_DEFAULT, help='Beta value for KL divergence.')
    exp_parser.add_argument('--experts_fusion', type=str, default='poe', choices=EXPERTS_FUSION_TYPES, help='Type of experts to use in the fusion of the modalities for the mvae.')
    exp_parser.add_argument('--rep_trick_mean', type=float, default=REPARAMETERIZATION_MEAN_DEFAULT, help='Mean value for the reparameterization trick for the vae and mvae.')
    exp_parser.add_argument('--rep_trick_std', type=float, default=REPARAMETERIZATION_STD_DEFAULT, help='Standard deviation value for the reparameterization trick for the vae and mvae.')
    exp_parser.add_argument('--poe_eps', type=float, default=POE_EPS_DEFAULT, help='Epsilon value for the product of experts fusion for the mvae.')
    exp_parser.add_argument('--adam_betas', nargs=2, type=float, default=ADAM_BETAS_DEFAULTS, help='Beta values for the Adam optimizer.')
    exp_parser.add_argument('--momentum', type=float, default=MOMENTUM_DEFAULT, help='Momentum for the SGD optimizer.')
    exp_parser.add_argument('--noise_mean', type=float, default=NOISE_MEAN_DEFAULT, help='Mean for noise distribution.')
    exp_parser.add_argument('--noise_std', type=float, default=NOISE_STD_DEFAULT, help='Standard deviation for noise distribution.')
    exp_parser.add_argument('--adv_epsilon', type=float, default=ADV_EPSILON_DEFAULT, help='Epsilon value for adversarial example generation.')
    exp_parser.add_argument('--download', type=bool, default=False, help='If true, downloads the choosen dataset.')
    
    args = vars(parser.parse_args())

    if args['command'] == 'clear':
        if args['clear_results']:
            shutil.rmtree(os.path.join(m_path, "results"), ignore_errors=True)
        if args['clear_checkpoints']:
            shutil.rmtree(os.path.join(m_path, "checkpoints"), ignore_errors=True)
        if args['clear_saved_models']:
            shutil.rmtree(os.path.join(m_path, "saved_models"), ignore_errors=True)
        if args['clear_wandb']:
            shutil.rmtree(os.path.join(m_path, "wandb"), ignore_errors=True)
        if args['clear_idx']:
            path = os.path.join(m_path, "experiments_idx.pickle")
            if os.path.exists(path):
                os.remove(path)
        if args['clear_configs']:
            for dir in os.listdir(os.path.join(m_path, "configs")):
                if os.path.isdir(os.path.join(m_path, "configs", dir)):
                    shutil.rmtree(os.path.join(m_path, "configs", dir), ignore_errors=True)
        sys.exit(0)
    
    torch.manual_seed(args['seed'])

    if args['command'] == 'config':
        if "config_permute" in args and args['config_permute'] is not None:
            conf_path = open(os.path.join(m_path, args['config_permute']))
            hyperparams = json.load(conf_path)
            keys, values = zip(*hyperparams.items())
            configs = [dict(zip(keys, v)) for v in itertools.product(*values)]
        else:
            config_data = json.load(open(os.path.join(m_path, args['load_config'])))
            configs = config_data['configs']
            if not isinstance(configs, list):
                configs = [configs]
            configs = [dict(item, **{'seed': args['seed']}) for item in configs]
        return configs
    
    if args['command'] == 'experiment':
        args.pop('command')
        return [args]

    raise argparse.ArgumentError("Argumment error: unknown command " + args['command'])

def setup_env(m_path, config):
    experiments_idx_path = os.path.join(m_path, "experiments_idx.pickle")
    if os.path.isfile(experiments_idx_path):
        with open(experiments_idx_path, 'rb') as idx_pickle:
            idx_dict = pickle.load(idx_pickle)
            idx_dict[config['stage']][config['dataset']][config['architecture']] += 1
            exp_id = idx_dict[config['stage']][config['dataset']][config['architecture']]
        with open(os.path.join(m_path, "experiments_idx.pickle"), "wb") as idx_pickle:
            pickle.dump(idx_dict, idx_pickle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(experiments_idx_path, 'wb') as idx_pickle:
            idx_dict = {}
            for stage in STAGES:
                idx_dict[stage] = {}
                for dataset in DATASETS:
                    idx_dict[stage][dataset] = dict.fromkeys(ARCHITECTURES, 0)

            idx_dict[config['stage']][config['dataset']][config['architecture']] = 1
            pickle.dump(idx_dict, idx_pickle, protocol=pickle.HIGHEST_PROTOCOL)
            exp_id = 1
    
    if 'model_out' not in config or config["model_out"] is None:
        model_out = config['architecture'] + '_' + config['dataset'] + f'_exp{exp_id}'
        if 'classifier' in config['stage']:
            model_out = 'clf_' + model_out
        config['model_out'] = model_out
    
    if 'config_out' not in config or config["config_out"] is None:
        config['config_out'] = config['model_out'] + '.json'

    config = config_validation(m_path, config)

    return config


def config_validation(m_path, config):
    if "stage" not in config or config["stage"] not in STAGES:
        raise argparse.ArgumentError("Argument error: must specify a valid pipeline stage.")
    if "architecture" not in config or config["architecture"] not in ARCHITECTURES:
        raise argparse.ArgumentError("Argument error: must specify an architecture for the experiments.")
    if "dataset" not in config or config["dataset"] not in DATASETS:
        raise argparse.ArgumentError("Argument error: must specify a dataset for the experiments.")
    
    try:
        os.makedirs(os.path.join(m_path, "configs", config['stage']), exist_ok=True)
        os.makedirs(os.path.join(m_path, "results", config['stage']), exist_ok=True)
    except IOError as e:
        traceback.print_exception(*sys.exc_info())
    finally:
        if config['stage'] == 'train_model' or config['stage'] == 'train_classifier':
            if config['stage'] == 'train_model' and config['model_out'] is None:
                raise argparse.ArgumentError('Argument error: the --model_out argument must be set when the --stage argument is ' + config['stage'] + '.')
            if config['epochs'] < 1:
                raise argparse.ArgumentError("Argument error: number of epochs must be a positive and non-zero integer.")
            elif config['batch_size'] < 1:
                raise argparse.ArgumentError("Argument error: batch_size value must be a positive and non-zero integer.")
            elif config['checkpoint'] < 0:
                raise argparse.ArgumentError("Argument error: checkpoint value must be an integer greater than or equal to 0.")
            elif config['checkpoint'] > config['epochs']:
                raise argparse.ArgumentError("Argument error: checkpoint value must be smaller than or equal to the number of epochs.")
    
        if "latent_dimension" not in config or config['latent_dimension'] is None:
            config['latent_dimension'] = LATENT_DIM_DEFAULT

        if config['latent_dimension'] < 1:
            raise argparse.ArgumentError("Argument error: latent_dimension value must be a positive and non-zero integer.")

        if "seed" not in config or config['seed'] is None:
            config["seed"] = SEED

        if "exclude_modality" not in config:
            config["exclude_modality"] = None

        if "target_modality" not in config:
            config['target_modality'] = None

        if config['exclude_modality'] is not None and config['target_modality'] is not None and config['exclude_modality'] == config['target_modality']:
            raise argparse.ArgumentError("Argument error: target modality cannot be the same as excluded modality.")

        if config['stage'] == 'train_model':
            if "load_config" in config and config["load_config"] is not None:
                config["load_config"] = None

        if "download" not in config:
            config["download"] = False

        with open(os.path.join(m_path, "results", config['stage'], config['model_out'] + ".txt"), 'w') as file:
            architecture = config['architecture']
            seed = config['seed']
            dataset = config['dataset']
            stage = config['stage']
            latent_dimension = config['latent_dimension']
            exclude_modality = config['exclude_modality']
            file.write(f'architecture: {architecture}\n')
            print(f'architecture: {architecture}')

            file.write(f'seed: {seed}\n')
            print(f'seed: {seed}')

            file.write(f'dataset: {dataset}\n')
            print(f'dataset: {dataset}')

            file.write(f'stage: {stage}\n')
            print(f'stage: {stage}')

            file.write(f'latent_dimension: {latent_dimension}\n')
            print(f'latent_dimension: {latent_dimension}')

            file.write(f'exclude_modality: {exclude_modality}\n')
            print(f'exclude_modality: {exclude_modality}')

            if stage == 'train_model' or stage == 'train_classifier':
                if "epochs" not in config:
                    config['epochs'] = EPOCHS_DEFAULT
                epochs = config['epochs']
                file.write(f'epochs: {epochs}\n')
                print(f'epochs: {epochs}')
            else:
                config['epochs'] = None

            if architecture == 'mvae':
                if "experts_fusion" not in config or config["experts_fusion"] is None:
                    config['experts_fusion'] = "poe"
                experts_fusion = config['experts_fusion'] 
                file.write(f'experts_fusion: {experts_fusion}\n')
                print(f'experts_fusion: {experts_fusion}')
                if experts_fusion == 'poe':
                    if "poe_eps" not in config or config["poe_eps"] is None:
                        config["poe_eps"] = POE_EPS_DEFAULT

                    poe_eps = config['poe_eps']
                    file.write(f'poe_eps: {poe_eps}\n')
                    print(f'poe_eps: {poe_eps}')
            else:
                config['experts_fusion'] = None
                config['poe_eps'] = None

            if architecture == 'vae' or architecture == 'mvae':
                if "rep_trick_mean" not in config:
                    config['rep_trick_mean'] = REPARAMETERIZATION_MEAN_DEFAULT
                if "rep_trick_std" not in config:
                    config['rep_trick_std'] = REPARAMETERIZATION_STD_DEFAULT

                rep_trick_mean = config['rep_trick_mean']
                rep_trick_std = config['rep_trick_std']
                file.write(f'rep_trick_mean: {rep_trick_mean}\n')
                print(f'rep_trick_mean: {rep_trick_mean}')
                file.write(f'rep_trick_std: {rep_trick_std}\n')
                print(f'rep_trick_std: {rep_trick_std}')
            else:
                config['rep_trick_mean'] = None
                config['rep_trick_std'] = None

            if "path_model" not in config:
                config["path_model"] = None

            if "path_model" not in config or config['path_model'] is not None:
                path_model = config['path_model']
                file.write(f'load_model_file: {path_model}\n')
                print(f'load_model_file: {path_model}')
                if stage == 'test_classifier':
                    if "path_classifier" not in config or config['path_classifier'] is None:
                        config['path_classifier'] = os.path.join(os.path.dirname(config['path_model']), "clf_" + os.path.basename(config['path_model']))
                    
                    path_clf = config['path_classifier']
                    file.write(f'load_classifier_file: {path_clf}\n')
                    print(f'load_classifier_file: {path_clf}')

            if "model_out" not in config:
                config['model_out'] = None

            if config['model_out'] is not None:
                path = config['model_out']
                if stage == 'train_model':
                    file.write(f'store_classifier_file: saved_models/{path}.pt\n')
                    print(f'store_model_file: saved_models/{path}.pt')
                elif stage == 'train_classifier':
                    file.write(f'store_classifier_file: saved_models/{path}.pt\n')
                    print(f'store_classifier_file: saved_models/{path}.pt')


            if stage == 'train_model' or stage == 'train_classifier' or stage == 'inference':
                checkpoint = config['checkpoint']
                file.write(f'checkpoint: {checkpoint}\n')
                print(f'checkpoint: {checkpoint}')
            

            if exclude_modality == 'image':
                config['image_recon_scale'] = 0.
                if config['target_modality'] == 'image':
                    config['target_modality'] = None
            elif exclude_modality == 'trajectory':
                config['traj_recon_scale'] = 0.
                if config['target_modality'] == 'trajectory':
                    config['target_modality'] = None

            if architecture == 'vae' or architecture == 'dae' or architecture == 'mvae':
                if "image_recon_scale" not in config:
                    config["image_recon_scale"] = RECON_SCALE_DEFAULTS['image']
                if "traj_recon_scale" not in config:
                    config["traj_recon_scale"] = RECON_SCALE_DEFAULTS['trajectory']
                img_scale = config['image_recon_scale']
                traj_recon_scale = config['traj_recon_scale']
                file.write(f'image_recon_scale: {img_scale}\n')
                print(f'image_recon_scale: {img_scale}')
                file.write(f'traj_recon_scale: {traj_recon_scale}\n')
                print(f'traj_recon_scale: {traj_recon_scale}')
                if architecture == 'vae' or architecture == 'mvae':
                    if "kld_beta" not in config:
                        config['kld_beta'] = KLD_BETA_DEFAULT
                    kld_beta = config['kld_beta']
                    file.write(f'kld_beta: {kld_beta}\n')
                    print(f'kld_beta: {kld_beta}')
            else:
                config['image_recon_scale'] = None
                config['traj_recon_scale'] = None
                config['kld_beta'] = None

            if architecture == 'gmc':
                if "infonce_temperature" not in config:
                    config["infonce_temperature"] = INFONCE_TEMPERATURE_DEFAULT
                temp = config['infonce_temperature']
                file.write(f'infonce_temperature: {temp}\n')
                print(f'infonce_temperature: {temp}')
            else:
                config['infonce_temperature'] = None

            if "noise" not in config:
                config["noise"] = None

            if config['noise'] is None:
                config["noise_mean"] = None
                config["noise_std"] = None

            if "adversarial_attack" not in config:
                config["adversarial_attack"] = None

            if config["adversarial_attack"] is None:
                config['adv_epsilon'] = None

            if "optimizer" not in config:
                config["optimizer"] = None
            
            if config["optimizer"] is None:
                config["learning_rate"] = None
                config["momentum"] = None
                config["adam_betas"] = None
            elif config["optimizer"] != 'sgd':
                config["momentum"] = None
            elif config["optimizer"] != 'adam':
                config["adam_betas"] = None

        return config


def setup_experiment(m_path, config, train=True):
    if torch.cuda.is_available():
        device = "cuda:0"
        print(f"Using device: {torch.cuda.get_device_name(0)}.")
        config['device'] = torch.cuda.get_device_name(0)
        with open(os.path.join(m_path, "results", config['stage'], config['model_out'] + ".txt"), 'r+') as file:
            content = file.read()
            file.seek(0, 0)
            file.write(f"Using device: {torch.cuda.get_device_name(0)}.\n" + content)
    else:
        device = "cpu"
        command = "cat /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True).decode().strip()
        device_info = ""
        for line in all_info.split("\n"):
            if "model name" in line:
                device_info = re.sub( ".*model name.*:", "", line,1)
                break

        print(f"Using device:{device_info}.")
        config['device'] = device_info
        with open(os.path.join(m_path, "results", config['stage'], config['model_out'] + ".txt"), 'r+') as file:
            content = file.read()
            file.seek(0, 0)
            file.write(f"Using device:{device_info}.\n" + content)

    device = torch.device(device)

    if config['dataset'] == 'mhd':
        dataset = MHDDataset(os.path.join(m_path, "datasets", "mhd"), device, config['download'], config['exclude_modality'], config['target_modality'], train)
    elif config['dataset'] == 'mosi':
        dataset = MOSIDataset(os.path.join(m_path, "datasets", "mosi"), device, config['download'], config['exclude_modality'], config['target_modality'], train)
    elif config['dataset'] == 'mosei':
        dataset = MOSEIDataset(os.path.join(m_path, "datasets", "mosei"), device, config['download'], config['exclude_modality'], config['target_modality'], train)
    elif config['dataset'] == 'pendulum':
        dataset = PendulumDataset(os.path.join(m_path, "datasets", "pendulum"), device, config['download'], config['exclude_modality'], config['target_modality'], train)
        

    if config['architecture'] == 'vae':
        scales = {'image': config['image_recon_scale'], 'trajectory': config['traj_recon_scale'], 'kld_beta': config['kld_beta']}
        model = vae.VAE(config['architecture'], config['latent_dimension'], device, config['exclude_modality'], scales, config['rep_trick_mean'], config['rep_trick_std'], dataset.dataset_len - dataset.dataset_len % config['batch_size'])
        loss_list_dict = {'elbo_loss': None, 'kld_loss': None, 'img_recon_loss': None, 'traj_recon_loss': None}
    elif config['architecture'] == 'dae':
        scales = {'image': config['image_recon_scale'], 'trajectory': config['traj_recon_scale']}
        model = dae.DAE(config['architecture'], config['latent_dimension'], device, config['exclude_modality'], scales)
        loss_list_dict = {'total_loss': None, 'img_recon_loss': None, 'traj_recon_loss': None}
    elif config['architecture'] == 'gmc':
        model = gmc.MhdGMC(config['architecture'], config['exclude_modality'], config['latent_dimension'])
        loss_list_dict = {'infonce_loss': None}
    elif config['architecture'] == 'mvae':
        scales = {'image': config['image_recon_scale'], 'trajectory': config['traj_recon_scale'], 'kld_beta': config['kld_beta']}
        model = mvae.MVAE(config['architecture'], config['latent_dimension'], device, config['exclude_modality'], scales, config['rep_trick_mean'], config['rep_trick_std'], config['experts_fusion'], config['poe_eps'], dataset.dataset_len - dataset.dataset_len % config['batch_size'])
        loss_list_dict = {'elbo_loss': None, 'kld_loss': None, 'img_recon_loss': None, 'traj_recon_loss': None}

    if "load_config" in config and config['load_config'] is not None:
        config_data = json.load(open(os.path.join(m_path, config['load_config'])))
        previous_config = config_data['configs']
        if "test" not in previous_config['stage'] and previous_config['stage'] != "inference":
            model.set_modalities(config['exclude_modality'])

    model.to(device)

    if 'classifier' in config['stage']:
        loss_list_dict = {'nll_loss': None, 'accuracy': None}
        model = classifier.MNISTClassifier(config['latent_dimension'], model)
        model.to(device)

    if config['adversarial_attack'] is not None or config['noise'] is not None:
        target_modality = config['target_modality']
        noise = config['noise']
        print(f'Target modality: {target_modality}')
        print(f'Noise: {noise}')
        with open(os.path.join(m_path, "results", config['stage'], config['model_out'] + ".txt"), 'a') as file:
            file.write(f'Target modality: {target_modality}\n')
            file.write(f'Noise: {noise}\n')
        
        if noise == "gaussian":
            transform = gaussian_noise.GaussianNoise(device, config['noise_mean'], config['noise_std'])
            noise_mean = config['noise_mean']
            noise_std = config['noise_std']
            
            print(f'gaussian_noise_mean: {noise_mean}')
            print(f'gaussian_noise_std: {noise_std}')
            with open(os.path.join(m_path, "results", config['stage'], config['model_out'] + ".txt"), 'a') as file:
                file.write(f'gaussian_noise_mean: {noise_mean}\n')
                file.write(f'gaussian_noise_std: {noise_std}\n')
        else:
            transform = None

        dataset._set_transform(transform)

        adversarial_attack = config['adversarial_attack']
        print(f'Adversarial attack: {adversarial_attack}')
        with open(os.path.join(m_path, "results", config['stage'], config['model_out'] + ".txt"), 'a') as file:
            file.write(f'Adversarial attack: {adversarial_attack}\n')

        if adversarial_attack == 'fgsm':
            fgsm_eps = config['adv_epsilon']
            adv_attack = fgsm.FGSM(device, model, target_modality, eps=fgsm_eps)

            print(f'fgsm_epsilon_value: {fgsm_eps}')
            with open(os.path.join(m_path, "results", config['model_out'] + ".txt"), 'a') as file:
                file.write(f'fgsm_epsilon: {fgsm_eps}\n')
        else:
            adv_attack = None

        dataset._set_adv_attack(adv_attack)

    if train:
        for key in loss_list_dict.keys():
            loss_list_dict[key] = np.zeros(config['epochs'])
        
        batch_size = config['batch_size']        
        batch_number = math.floor(dataset.dataset_len/batch_size)

        print(f'batch_size: {batch_size}')
        print(f'number_batches: {batch_number}')
        with open(os.path.join(m_path, "results", config['stage'], config['model_out'] + ".txt"), 'a') as file:
            file.write(f'batch_size: {batch_size}\n')
            file.write(f'number_batches: {batch_number}\n')
    
        if config['optimizer'] is not None:
            optimizer_str = config['optimizer']
            print(f'optimizer: {optimizer_str}')

            lr = config['learning_rate']
            print(f'learning_rate: {lr}')
            with open(os.path.join(m_path, "results", config['stage'], config['model_out'] + ".txt"), 'a') as file:
                file.write(f'optimizer: {optimizer_str}\n')
                file.write(f'learning_rate: {lr}\n')

            if optimizer_str == 'adam':
                betas = config['adam_betas']
                optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas)
                print(f'adam_betas: {optimizer_str}')
                with open(os.path.join(m_path, "results", config['stage'], config['model_out'] + ".txt"), 'a') as file:
                    file.write(f'adam_betas: {betas}\n')
            elif optimizer_str == 'sgd':
                momentum = config['momentum']
                optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
                print(f'sgd_momentum: {momentum}')
                with open(os.path.join(m_path, "results", config['stage'], config['model_out'] + ".txt"), 'a') as file:
                    file.write(f'sgd_momentum: {momentum}\n')
    else:
        optimizer = None
        batch_number = None
        for key in loss_list_dict.keys():
            loss_list_dict[key] = 0.


    if "notes" not in config or config["notes"] is None:
        print('Enter experiment notes:')
        notes, _, _ = select.select([sys.stdin], [], [], TIMEOUT)
        if (notes):
            notes = sys.stdin.readline().strip()
        else:
            notes = ""
        termios.tcflush(sys.stdin, termios.TCIOFLUSH)
    else:
        notes = config["notes"]

    wandb.init(project="rmgm", 
               name=config['model_out'],
               config={key: value for key, value in config.items() if value is not None}, 
               notes=notes,
               allow_val_change=True,
               magic=True,
               #mode="offline",
               tags=[config['architecture'], config['dataset'], config['stage']])
    wandb.watch(model)

    return device, dataset, model, loss_list_dict, batch_number, optimizer