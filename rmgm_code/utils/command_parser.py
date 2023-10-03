import re
import os
import sys
import json
import torch
import wandb
import select
import shutil
#import termios
import argparse
import threading
import itertools
import traceback
import subprocess

import torch.optim as optim

from utils.logger import plot_loss_compare_graph, plot_metric_compare_bar, plot_bar_across_seeds, plot_graph_across_seeds
from input_transformations import gaussian_noise, fgsm, pgd, cw
from architectures.mhd.downstream.classifier import MHDClassifier
from architectures.mhd.models.vae import MhdVAE
from architectures.mhd.models.dae import MhdDAE
from architectures.mhd.models.mvae import MhdMVAE
from architectures.mhd.models.gmc import MhdGMC
from architectures.mhd.models.dgmc import MhdDGMC
from architectures.mhd.models.rgmc import MhdRGMC
from architectures.mhd.models.gmcwd import MhdGMCWD
from architectures.mnist_svhn.downstream.classifier import MSClassifier
from architectures.mnist_svhn.models.vae import MSVAE
from architectures.mnist_svhn.models.dae import MSDAE
from architectures.mnist_svhn.models.mvae import MSMVAE
from architectures.mnist_svhn.models.gmc import MSGMC
from architectures.mnist_svhn.models.dgmc import MSDGMC
from architectures.mnist_svhn.models.rgmc import MSRGMC
from architectures.mnist_svhn.models.gmcwd import MSGMCWD
from architectures.pendulum.models.vae import PendulumVAE
from architectures.pendulum.models.dae import PendulumDAE
from architectures.pendulum.models.mvae import PendulumMVAE
from architectures.pendulum.models.gmc import PendulumGMC
from architectures.pendulum.models.dgmc import PendulumDGMC
from architectures.pendulum.models.gmcwd import PendulumGMCWD
from architectures.pendulum.models.rgmc import PendulumRGMC
from datasets.mhd.mhd_dataset import MhdDataset
from datasets.mosi.mosi_dataset import MosiDataset
from datasets.mosei.mosi_dataset import MoseiDataset
from datasets.pendulum.pendulum_dataset import PendulumDataset
from datasets.mnist_svhn.mnist_svhn_dataset import MnistSvhnDataset

TIMEOUT = 0 # Seconds to wait for user to input notes
ARCHITECTURES = ['vae', 'dae', 'mvae', 'gmc', 'dgmc', 'rgmc', 'gmcwd', None]
DATASETS = ['mhd', 'mnist_svhn', 'mosi', 'mosei', 'pendulum']
OPTIMIZERS = ['sgd', 'adam', None]
ADVERSARIAL_ATTACKS = ["gaussian_noise", "fgsm", "pgd", None]
EXPERTS_FUSION_TYPES = ['poe', 'moe', None]
MODALITIES = {
    'mhd': ['image', 'trajectory'],
    'mnist_svhn': ['mnist', 'svhn'],
    'pendulum': ['image', 'sound']
    }
STAGES = ['train_model', 'train_classifier', 'test_model', 'test_classifier', 'inference']

SEED = 42
LR_DEFAULT = 0.001
EPOCHS_DEFAULT = 100
BATCH_SIZE_DEFAULT = 64
CHECKPOINT_DEFAULT = 0
LATENT_DIM_DEFAULT = 64
COMMON_DIM_DEFAULT = 64
INFONCE_TEMPERATURE_DEFAULT = 0.1
RECON_SCALE_DEFAULTS = {'image': 0.5, 'trajectory': 0.5, 'mnist': 0.5, 'svhn': 0.5}
KLD_BETA_DEFAULT = 0.5
REPARAMETERIZATION_MEAN_DEFAULT = 0.0
REPARAMETERIZATION_STD_DEFAULT = 1.0
EXPERTS_FUSION_DEFAULT = "poe"
POE_EPS_DEFAULT = 1e-8
O3N_LOSS_SCALE_DEFAULT = 1.0
MODEL_TRAIN_NOISE_FACTOR_DEFAULT = 1.0
MOMENTUM_DEFAULT = 0.9
ADAM_BETAS_DEFAULTS = [0.9, 0.999]
NOISE_STD_DEFAULT = 1.0
ADV_EPSILON_DEFAULT = 8 / 255
ADV_ALPHA_DEFAULT = 2 / 255
ADV_STEPS_DEFAULT = 10

idx_lock = threading.Lock()
device_lock = threading.Lock()

def process_arguments(m_path):
    parser = argparse.ArgumentParser(prog="rmgm", description="Program tests the performance and robustness of several generative models with clean and noisy/adversarial samples.")
    subparsers = parser.add_subparsers(help="command", dest="command")
    comp_parser = subparsers.add_parser("compare")
    comp_parser.add_argument('-a', '--architecture', choices=ARCHITECTURES, help='Architecture to be used in the comparison.')
    comp_parser.add_argument('-d', '--dataset', type=str, default='mhd', choices=DATASETS, help='Dataset to be used in the comparison.')
    comp_parser.add_argument('-s', '--stage', type=str, default='train_model', choices=STAGES, help='Stage of the pipeline to be used in the comparison.')
    comp_parser.add_argument('--model_outs', '--mos', nargs='+')
    comp_parser.add_argument('--param_comp', '--pc', type=str)
    comp_parser.add_argument('--parent_param', '--pp', type=str)
    comp_parser.add_argument('--number_seeds', '--ns', type=int)
    comp_parser.add_argument('--target_modality', '--tm', type=str)

    upload_parser = subparsers.add_parser("upload")
    upload_parser.add_argument('--configs')
    upload_parser.add_argument('--saved_models')
    upload_parser.add_argument('--results')
    upload_parser.add_argument('--checkpoints')

    clear_parser = subparsers.add_parser("clear")
    clear_parser.add_argument('--clear_results', '--clear_res', action="store_false", help="Flag to delete results directory.")
    clear_parser.add_argument('--clear_checkpoints', '--clear_check', action="store_false", help="Flag to delete checkpoints directory.")
    clear_parser.add_argument('--clear_saved_models', '--clear_models', '--clear_saved', action="store_false", help="Flag to delete saved_models directory.")
    clear_parser.add_argument('--clear_wandb', '--clear_w&b', action="store_false", help="Flag to delete wandb directory.")
    clear_parser.add_argument('--clear_configs', '--clear_runs', action="store_false", help="Flag to delete configs directory.")
    clear_parser.add_argument('--clear_idx', action='store_false', help="Flag to delete previous experiments idx file.")

    configs_parser = subparsers.add_parser("config")
    configs_parser.add_argument('--load_config', '--load_json', type=str, help='File path where the experiment(s) configurations are to loaded from.')
    configs_parser.add_argument('--config_permute', '--permute_config', type=str, help='Generate several config runs from permutations of dict of lists with hyperparams.')
    configs_parser.add_argument('--seed', '--torch_seed', type=int, default=SEED, help='Seed value for results replication.')

    exp_parser = subparsers.add_parser("experiment")
    exp_parser.add_argument('-a', '--architecture', choices=ARCHITECTURES, help='Architecture to be used in the experiment.')
    exp_parser.add_argument('-p', '--path_model', type=str, default=None, help="Filename of the file where the model is to be loaded from.")
    exp_parser.add_argument('--seed', '--torch_seed', '--pytorch_seed', type=int, default=SEED, help='Seed value for results replication.')
    exp_parser.add_argument('--path_classifier', type=str, default=None, help="Filename of the file where the classifier is to be loaded from.")
    exp_parser.add_argument('-m', '--model_out', type=str, default=None, help="Filename of the file where the model/classifier and results are to be saved to.")
    exp_parser.add_argument('-d', '--dataset', type=str, default='mhd', choices=DATASETS, help='Dataset to be used in the experiments.')
    exp_parser.add_argument('-s', '--stage', type=str, default='train_model', choices=STAGES, help='Stage of the pipeline to be executed in the experiment.')
    exp_parser.add_argument('-o', '--optimizer', type=str, default='sgd', choices=OPTIMIZERS, help='Optimizer for the model training process.')
    exp_parser.add_argument('-r', '--learning_rate', '--lr', type=float, default=LR_DEFAULT, help='Learning rate value for the optimizer.')
    exp_parser.add_argument('-e', '--epochs', type=int, default=EPOCHS_DEFAULT, help='Number of epochs to train the model.')
    exp_parser.add_argument('-b', '--batch_size', type=int, default=BATCH_SIZE_DEFAULT, help='Number of samples processed for each model update.')
    exp_parser.add_argument('--checkpoint', type=int, default=CHECKPOINT_DEFAULT, help='Epoch interval between checkpoints of the model in training.')
    exp_parser.add_argument('--latent_dimension', '--latent_dim', type=int, default=LATENT_DIM_DEFAULT, help='Dimension of the latent space of the models encodings.')
    exp_parser.add_argument('--common_dimension', '--common_dim', type=int, default=COMMON_DIM_DEFAULT, help='Dimension of the common representation space of the models based on GMC.')
    exp_parser.add_argument('--adversarial_attack', '--attack', type=str, default=None, choices=ADVERSARIAL_ATTACKS, help='Execute an adversarial attack against the model.')
    exp_parser.add_argument('--target_modality', type=str, default=None, help='Modality to target with noisy and/or adversarial samples.')
    exp_parser.add_argument('--exclude_modality', type=str, default=None, help='Exclude a modality from the training/testing process.')
    exp_parser.add_argument('--infonce_temperature', '--infonce_temp', type=float, default=INFONCE_TEMPERATURE_DEFAULT, help='Temperature for the infonce loss.')
    exp_parser.add_argument('--image_recon_scale', type=float, default=RECON_SCALE_DEFAULTS['image'], help='Weight for the image reconstruction loss.')
    exp_parser.add_argument('--traj_recon_scale', type=float, default=RECON_SCALE_DEFAULTS['trajectory'], help='Weight for the trajectory reconstruction loss.')
    exp_parser.add_argument('--kld_beta', type=float, default=KLD_BETA_DEFAULT, help='Beta value for KL divergence.')
    exp_parser.add_argument('--experts_fusion', type=str, default='poe', choices=EXPERTS_FUSION_TYPES, help='Type of experts to use in the fusion of the modalities for the mvae.')
    exp_parser.add_argument('--rep_trick_mean', type=float, default=REPARAMETERIZATION_MEAN_DEFAULT, help='Mean value for the reparameterization trick for the vae and mvae.')
    exp_parser.add_argument('--rep_trick_std', type=float, default=REPARAMETERIZATION_STD_DEFAULT, help='Standard deviation value for the reparameterization trick for the vae and mvae.')
    exp_parser.add_argument('--poe_eps', type=float, default=POE_EPS_DEFAULT, help='Epsilon value for the product of experts fusion for the mvae.')
    exp_parser.add_argument('--train_noise_factor', type=float, default=MODEL_TRAIN_NOISE_FACTOR_DEFAULT)
    exp_parser.add_argument('--adam_betas', nargs=2, type=float, default=ADAM_BETAS_DEFAULTS, help='Beta values for the Adam optimizer.')
    exp_parser.add_argument('--momentum', type=float, default=MOMENTUM_DEFAULT, help='Momentum for the SGD optimizer.')
    exp_parser.add_argument('--noise_std', type=float, default=NOISE_STD_DEFAULT, help='Standard deviation for noise distribution.')
    exp_parser.add_argument('--adv_epsilon', type=float, default=ADV_EPSILON_DEFAULT, help='Epsilon value for adversarial example generation.')
    exp_parser.add_argument('--download', type=bool, default=False, help='If true, downloads the choosen dataset.')
    
    args = vars(parser.parse_args())
    if args['command'] == 'compare':
        config = {
            'architecture': args['architecture'],
            'dataset': args['dataset'],
            'stage': args['stage'],
            'model_outs': args['model_outs'],
            'param_comp': args['param_comp'],
        }
        if 'parent_param' in args:
            config['parent_param'] = args['parent_param']

        metrics_analysis(m_path, config)
        sys.exit(0)

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
            path = os.path.join(m_path, "experiments_idx.json")
            device_idx_path = os.path.join(m_path, "device_idx.txt")
            if os.path.exists(path):
                os.remove(path)
            if os.path.exists(device_idx_path):
                os.remove(device_idx_path)
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
    experiments_idx_path = os.path.join(m_path, "experiments_idx.json")
    idx_dict = {}
    idx_lock.acquire()
    if os.path.isfile(experiments_idx_path):
        with open(experiments_idx_path, 'r') as idx_json:
            idx_dict = json.load(idx_json)
            idx_dict[config['stage']][config['dataset']][config['architecture']] += 1
            exp_id = idx_dict[config['stage']][config['dataset']][config['architecture']]
        with open(experiments_idx_path, 'w') as idx_json:
            json.dump(idx_dict, idx_json, indent=4)
    else:
        with open(experiments_idx_path, 'w') as idx_json:
            idx_dict = create_idx_dict()
            idx_dict[config['stage']][config['dataset']][config['architecture']] = 1
            json.dump(idx_dict, idx_json, indent=4)
            exp_id = 1
    idx_lock.release()
    
    if 'model_out' not in config or config["model_out"] is None:
        model_out = config['architecture'] + '_' + config['dataset'] + f'_exp{exp_id}'
        if 'classifier' in config['stage']:
            model_out = 'clf_' + model_out
        config['model_out'] = model_out

    if ("path_model" not in config or config["path_model"] is None) and config["stage"] != "train_model":
        config["path_model"] = os.path.join("saved_models", config["architecture"] + "_" + config["dataset"] + f"_exp{exp_id}.pt")

    if ("path_classifier" not in config or config["path_classifier"] is None) and config["stage"] == "test_classifier":
        config["path_classifier"] = os.path.join("saved_models", "clf_" + os.path.basename(config["path_model"]))

    config = config_validation(m_path, config)

    return config

def create_idx_dict():
    idx_dict = {}
    for stage in STAGES:
        idx_dict[stage] = {}
        for dataset in DATASETS:
            idx_dict[stage][dataset] = dict.fromkeys(ARCHITECTURES, 0)
    return idx_dict

def base_validation(function):
    def wrapper(m_path, config):
        if "stage" not in config or config["stage"] not in STAGES:
            raise argparse.ArgumentError("Argument error: must specify a valid pipeline stage.")
        if "architecture" not in config or config["architecture"] not in ARCHITECTURES:
            raise argparse.ArgumentError("Argument error: must specify an architecture for the experiments.")
        if "dataset" not in config or config["dataset"] not in DATASETS:
            raise argparse.ArgumentError("Argument error: must specify a dataset for the experiments.")
        return function(m_path, config)
    return wrapper

@base_validation
def config_validation(m_path, config):
    if "batch_size" not in config or config["batch_size"] is None:
            config['batch_size'] = BATCH_SIZE_DEFAULT
    elif config['batch_size'] < 1:
        raise argparse.ArgumentError("Argument error: batch_size value must be a positive and non-zero integer.")
    if "latent_dimension" not in config or config['latent_dimension'] is None:
        config['latent_dimension'] = LATENT_DIM_DEFAULT
    elif config['latent_dimension'] < 1:
        raise argparse.ArgumentError("Argument error: latent_dimension value must be a positive and non-zero integer.")
    
    if config['architecture'] is None:
        raise argparse.ArgumentError("Argument error: must define a valid architecture.")
    if "exclude_modality" in config and config["exclude_modality"] not in MODALITIES[config['dataset']]:
        raise argparse.ArgumentError("Argument error: must define a valid modality to exclude.")
    if "adversarial_attack" in config and config['adversarial_attack'] is not None:
        if config["adversarial_attack"] not in ADVERSARIAL_ATTACKS:
            raise argparse.ArgumentError("Argument error: must define valid adversarial attack.")
        if "target_modality" not in config or config['target_modality'] not in MODALITIES[config['dataset']]:
            raise argparse.ArgumentError("Argument error: must specify valid target_modality for adversarial attack.")
    else:
        config["target_modality"] = None

    if ("exclude_modality" in config and config['exclude_modality'] is not None) and config['target_modality'] is not None and config['exclude_modality'] == config['target_modality']:
        raise argparse.ArgumentError("Argument error: target modality cannot be the same as excluded modality.")
    
    try:
        os.makedirs(os.path.join(m_path, "results", config['stage']), exist_ok=True)
        if config['stage'] == 'train_model' or config['stage'] == 'train_classifier':
            os.makedirs(os.path.join(m_path, "configs", config['stage']), exist_ok=True)
            if "epochs" not in config or config["epochs"] is None:
                config['epochs'] = EPOCHS_DEFAULT
                
            if config['checkpoint'] < 0:
                raise argparse.ArgumentError("Argument error: checkpoint value must be an integer greater than or equal to 0.")
            elif config['checkpoint'] > config['epochs']:
                raise argparse.ArgumentError("Argument error: checkpoint value must be smaller than or equal to the number of epochs.")
        else:
            if "epochs" in config and config["epochs"] is not None:
                config["epochs"] = None
            if "learning_rate" in config and config['learning_rate'] is not None:
                config['learning_rate'] = None
            if "optimizer" in config and config['optimizer'] is not None:
                config['optimizer'] = None
            if config["stage"] != "inference" and "checkpoint" in config and config["checkpoint"] is not None:
                config["checkpoint"] = None
    except IOError as e:
        traceback.print_exception(*sys.exc_info())
    finally:
        if "seed" not in config or config['seed'] is None:
            config["seed"] = SEED

        if "exclude_modality" not in config:
            config["exclude_modality"] = None

        if "target_modality" not in config:
            config['target_modality'] = None

        if "download" not in config:
            config["download"] = None

        if "ae" in config['architecture'] or config['architecture'] == "dgmc" or config['architecture'] == 'gmcwd':
            if config['dataset'] == "mhd":
                if "image_recon_scale" not in config:
                    config["image_recon_scale"] = RECON_SCALE_DEFAULTS['image']
                if "traj_recon_scale" not in config:
                    config["traj_recon_scale"] = RECON_SCALE_DEFAULTS['trajectory']
            elif config['dataset'] == "mnist_svhn":
                if "mnist_recon_scale" not in config:
                    config["mnist_recon_scale"] = RECON_SCALE_DEFAULTS['mnist']
                if "svhn_recon_scale" not in config:
                    config["svhn_recon_scale"] = RECON_SCALE_DEFAULTS['svhn']

            if config['architecture'] == 'dae' or config['architecture'] == 'dgmc' or config['architecture'] == 'gmcwd':
                if "train_noise_factor" not in config or config['train_noise_factor'] is None:
                    config['train_noise_factor'] = MODEL_TRAIN_NOISE_FACTOR_DEFAULT

            if "vae" in config['architecture']:
                if "kld_beta" not in config:
                    config['kld_beta'] = KLD_BETA_DEFAULT
                if "rep_trick_mean" not in config or config['rep_trick_mean'] is None:
                    config['rep_trick_mean'] = REPARAMETERIZATION_MEAN_DEFAULT
                if "rep_trick_std" not in config or config['rep_trick_std'] is None:
                    config['rep_trick_std'] = REPARAMETERIZATION_STD_DEFAULT

                if config['architecture'] == 'mvae':
                    if "experts_fusion" not in config or config["experts_fusion"] is None:
                        config['experts_fusion'] = EXPERTS_FUSION_DEFAULT

                    if config['experts_fusion'] == 'poe':
                        if "poe_eps" not in config or config["poe_eps"] is None:
                            config["poe_eps"] = POE_EPS_DEFAULT
                else:
                    config['experts_fusion'] = None
                    config['poe_eps'] = None
            else:
                config['kld_beta'] = None
                config['rep_trick_mean'] = None
                config['rep_trick_std'] = None

            if config['dataset'] == 'mhd':
                if config['exclude_modality'] == 'image':
                    config['image_recon_scale'] = 0.
                elif config['exclude_modality'] == 'trajectory':
                    config['traj_recon_scale'] = 0.
            elif config['dataset'] == 'mnist_svhn':
                if config['exclude_modality'] == 'mnist':
                    config['mnist_recon_scale'] = 0.
                elif config['exclude_modality'] == 'svhn':
                    config['svhn_recon_scale'] = 0.
        else:
            config['image_recon_scale'] = None
            config['traj_recon_scale'] = None
            config['mnist_recon_scale'] = None
            config['svhn_recon_scale'] = None
            config['kld_beta'] = None
            config['rep_trick_mean'] = None
            config['rep_trick_std'] = None
            config['experts_fusion'] = None
            config['poe_eps'] = None
            
        if "gmc" in config['architecture']:
            if "infonce_temperature" not in config:
                config["infonce_temperature"] = INFONCE_TEMPERATURE_DEFAULT

            if "common_dimension" not in config or config['common_dimension'] is None:
                config['common_dimension'] = COMMON_DIM_DEFAULT
            
            if config['architecture'] == 'rgmc':
                if "o3n_loss_scale" not in config:
                    config['o3n_loss_scale'] = O3N_LOSS_SCALE_DEFAULT
        else:
            config['infonce_temperature'] = None
            config['common_dimension'] = None

        if config['stage'] == "train_model":
            config["path_model"] = None

        if "model" in config['stage']:
            config["path_classifier"] = None
        else:
            config['image_recon_scale'] = None
            config['traj_recon_scale'] = None
            config['mnist_recon_scale'] = None
            config['svhn_recon_scale'] = None
            config['infonce_temperature'] = None
            config['o3n_loss_scale'] = None
            config['kld_beta'] = None
        
        if config['stage'] == 'test_classifier':
            if "path_classifier" not in config or config['path_classifier'] is None:
                config['path_classifier'] = os.path.join(os.path.dirname(config['path_model']), "clf_" + os.path.basename(config['path_model']))

        if "adversarial_attack" not in config:
            config["adversarial_attack"] = None

        if config["adversarial_attack"] is None:
            config["noise_std"] = None
            config['adv_epsilon'] = None
            config['adv_alpha'] = None
            config['adv_steps'] = None
        else:
            if config["adversarial_attack"] == 'gaussian_noise':
                if "noise_std" not in config or config["noise_std"] is None:
                    config["noise_std"] = NOISE_STD_DEFAULT
            if config["adversarial_attack"] == 'fgsm' or config["adversarial_attack"] == 'pgd':
                if 'adv_epsilon' not in config or config['adv_epsilon'] is None:
                    config['adv_epsilon'] = ADV_EPSILON_DEFAULT
                    
                if config["adversarial_attack"] == 'pgd':
                    if 'adv_alpha' not in config or config['adv_alpha'] is None:
                        config['adv_alpha'] = ADV_ALPHA_DEFAULT
                    if 'adv_steps' not in config or config['adv_steps'] is None:
                        config['adv_steps'] = ADV_STEPS_DEFAULT

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

def setup_device(m_path, config):
    if torch.cuda.is_available():
        device_idx_path = os.path.join(m_path, "device_idx.txt")
        device_lock.acquire()
        if os.path.isfile(device_idx_path):
            with open(device_idx_path, 'r+') as device_file:
                device_counter = int(device_file.readline()) + 1
                device_file.seek(0)
                device_file.truncate()
                device_file.write(str(device_counter))
        else:
            with open(device_idx_path, 'w') as device_file:
                device_counter = 0
                device_file.write('0')
        
        device_id = device_counter % torch.cuda.device_count()
        device = f"cuda:{device_id}"
        device_lock.release()
        config['device'] = torch.cuda.get_device_name(torch.cuda.current_device())
    else:
        device = "cpu"
        command = "cat /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True).decode().strip()
        device_info = ""
        for line in all_info.split("\n"):
            if "model name" in line:
                device_info = re.sub( ".*model name.*:", "", line, 1)
                break

        config['device'] = device_info

    return device

def setup_experiment(m_path, config, device, train=True):
    def setup_dataset(m_path, config, device, train):
        if config['dataset'] == 'mhd':
            dataset = MhdDataset('mhd', os.path.join(m_path, "datasets", "mhd"), device, config['download'], config['exclude_modality'], config['target_modality'], train)
        elif config['dataset'] == 'mosi':
            dataset = MosiDataset('mosi', os.path.join(m_path, "datasets", "mosi"), device, config['download'], config['exclude_modality'], config['target_modality'], train)
        elif config['dataset'] == 'mosei':
            dataset = MoseiDataset('mosei', os.path.join(m_path, "datasets", "mosei"), device, config['download'], config['exclude_modality'], config['target_modality'], train)
        elif config['dataset'] == 'pendulum':
            dataset = PendulumDataset('pendulum', os.path.join(m_path, "datasets", "pendulum"), device, config['download'], config['exclude_modality'], config['target_modality'], train)
        elif config['dataset'] == 'mnist_svhn':
            dataset = MnistSvhnDataset('mnist_svhn', os.path.join(m_path, "datasets", "mnist_svhn"), device, config['download'], config['exclude_modality'], config['target_modality'], train)
        return dataset
    
    def setup_classifier(model, latent_dim, exclude_mod):
        if config['dataset'] == 'mhd':
            clf = MHDClassifier(latent_dim, model, exclude_mod)
        elif config['dataset'] == 'mnist_svhn':
            clf = MSClassifier(latent_dim, model, exclude_mod)
        return clf

    if config['stage'] == "inference":
        dataset = setup_dataset(m_path, config, device, True)
    else:
        dataset = setup_dataset(m_path, config, device, train)

    if "download" in config:
        config['download'] = None

    if config['stage'] != "train_model":
        model_config = json.load(open(os.path.join(m_path, "configs", "train_model", os.path.basename(os.path.splitext(config['path_model'])[0]) + '.json')))
        latent_dim = model_config["latent_dimension"]
        exclude_modality = model_config["exclude_modality"]
    else:
        latent_dim = config["latent_dimension"]
        exclude_modality = config["exclude_modality"]

    if config['dataset'] == 'mhd':
        if config['architecture'] == 'vae':
            scales = {'image': config['image_recon_scale'], 'trajectory': config['traj_recon_scale'], 'kld_beta': config['kld_beta']}
            model = MhdVAE(config['architecture'], latent_dim, device, exclude_modality, scales, config['rep_trick_mean'], config['rep_trick_std'])
        elif config['architecture'] == 'dae':
            scales = {'image': config['image_recon_scale'], 'trajectory': config['traj_recon_scale']}
            model = MhdDAE(config['architecture'], latent_dim, device, exclude_modality, scales, noise_factor=config['train_noise_factor'])
        elif config['architecture'] == 'mvae':
            scales = {'image': config['image_recon_scale'], 'trajectory': config['traj_recon_scale'], 'kld_beta': config['kld_beta']}
            model = MhdMVAE(config['architecture'], latent_dim, device, exclude_modality, scales, config['rep_trick_mean'], config['rep_trick_std'], config['experts_fusion'], config['poe_eps'])
        elif config['architecture'] == 'gmc':
            model = MhdGMC(config['architecture'], exclude_modality, config['common_dimension'], latent_dim, config['infonce_temperature'])
        elif config['architecture'] == 'dgmc':
            scales = {'image': config['image_recon_scale'], 'trajectory': config['traj_recon_scale'], 'infonce_temp': config['infonce_temperature']}
            model = MhdDGMC(config['architecture'], exclude_modality, config['common_dimension'], latent_dim, scales, noise_factor=config['train_noise_factor'])
        elif config['architecture'] == 'rgmc':
            scales = {'infonce_temp': config['infonce_temperature'], 'o3n_loss': config['o3n_loss_scale']}
            model = MhdRGMC(config['architecture'], exclude_modality, config['common_dimension'], latent_dim, scales, noise_factor=config['train_noise_factor'], device=device)
        elif config['architecture'] == 'gmcwd':
            scales = {'image': config['image_recon_scale'], 'trajectory': config['traj_recon_scale'], 'infonce_temp': config['infonce_temperature']}
            model = MhdGMCWD(config['architecture'], exclude_modality, config['common_dimension'], latent_dim, scales, noise_factor=config['train_noise_factor'])
    elif config['dataset'] == 'mnist_svhn':
        if config['architecture'] == 'vae':
            scales = {'mnist': config['mnist_recon_scale'], 'svhn': config['svhn_recon_scale'], 'kld_beta': config['kld_beta']}
            model = MSVAE(config['architecture'], latent_dim, device, exclude_modality, scales, config['rep_trick_mean'], config['rep_trick_std'])
        elif config['architecture'] == 'dae':
            scales = {'mnist': config['mnist_recon_scale'], 'svhn': config['svhn_recon_scale']}
            model = MSDAE(config['architecture'], latent_dim, device, exclude_modality, scales, noise_factor=config['train_noise_factor'])
        elif config['architecture'] == 'mvae':
            scales = {'mnist': config['mnist_recon_scale'], 'svhn': config['svhn_recon_scale'], 'kld_beta': config['kld_beta']}
            model = MSMVAE(config['architecture'], latent_dim, device, exclude_modality, scales, config['rep_trick_mean'], config['rep_trick_std'], config['experts_fusion'], config['poe_eps'])
        elif config['architecture'] == 'gmc':
            model = MSGMC(config['architecture'], exclude_modality, config['common_dimension'], latent_dim, config['infonce_temperature'])
        elif config['architecture'] == 'dgmc':
            scales = {'mnist': config['mnist_recon_scale'], 'svhn': config['svhn_recon_scale'], 'infonce_temp': config['infonce_temperature']}
            model = MSDGMC(config['architecture'], exclude_modality, config['common_dimension'], latent_dim, scales, noise_factor=config['train_noise_factor'])
        elif config['architecture'] == 'rgmc':
            scales = {'infonce_temp': config['infonce_temperature'], 'o3n_loss': config['o3n_loss_scale']}
            model = MSRGMC(config['architecture'], exclude_modality, config['common_dimension'], latent_dim, scales, noise_factor=config['train_noise_factor'], device=device)
        elif config['architecture'] == 'gmcwd':
            scales = {'mnist': config['mnist_recon_scale'], 'svhn': config['svhn_recon_scale'], 'infonce_temp': config['infonce_temperature']}
            model = MSGMCWD(config['architecture'], exclude_modality, config['common_dimension'], latent_dim, scales, noise_factor=config['train_noise_factor'])
    elif config['dataset'] == 'pendulum':
        if config['architecture'] == 'vae':
            scales = {'image': config['image_recon_scale'], 'sound': config['sound_recon_scale'], 'kld_beta': config['kld_beta']}
            model = PendulumVAE(config['architecture'], latent_dim, device, exclude_modality, scales, config['rep_trick_mean'], config['rep_trick_std'])
        elif config['architecture'] == 'dae':
            scales = {'image': config['image_recon_scale'], 'sound': config['sound_recon_scale']}
            model = PendulumDAE(config['architecture'], latent_dim, device, exclude_modality, scales, noise_factor=config['train_noise_factor'])
        elif config['architecture'] == 'mvae':
            scales = {'image': config['image_recon_scale'], 'sound': config['sound_recon_scale'], 'kld_beta': config['kld_beta']}
            model = PendulumMVAE(config['architecture'], latent_dim, device, exclude_modality, scales, config['rep_trick_mean'], config['rep_trick_std'], config['experts_fusion'], config['poe_eps'])
        elif config['architecture'] == 'gmc':
            model = PendulumGMC(config['architecture'], exclude_modality, config['common_dimension'], latent_dim, config['infonce_temperature'])
        elif config['architecture'] == 'dgmc':
            scales = {'image': config['image_recon_scale'], 'sound': config['sound_recon_scale'], 'infonce_temp': config['infonce_temperature']}
            model = PendulumDGMC(config['architecture'], exclude_modality, config['common_dimension'], latent_dim, scales, noise_factor=config['train_noise_factor'])
        elif config['architecture'] == 'rgmc':
            scales = {'infonce_temp': config['infonce_temperature'], 'o3n_loss': config['o3n_loss_scale']}
            model = PendulumRGMC(config['architecture'], exclude_modality, config['common_dimension'], latent_dim, scales, noise_factor=config['train_noise_factor'], device=device)
        elif config['architecture'] == 'gmcwd':
            scales = {'image': config['image_recon_scale'], 'sound': config['sound_recon_scale'], 'infonce_temp': config['infonce_temperature']}
            model = PendulumGMCWD(config['architecture'], exclude_modality, config['common_dimension'], latent_dim, scales, noise_factor=config['train_noise_factor'])

    if "path_model" in config and config["path_model"] is not None and config["stage"] != "train_model":
        model.load_state_dict(torch.load(os.path.join(m_path, config["path_model"])))
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    model.to(device)
    if 'classifier' in config['stage']:
        if "path_classifier" in config and config["path_classifier"] is not None and config["stage"] == "test_classifier":
            clf_config = json.load(open(os.path.join(m_path, "configs", "train_classifier", os.path.basename(os.path.splitext(config['path_classifier'])[0]) + '.json')))
            model = setup_classifier(latent_dim=clf_config['latent_dimension'], model=model, exclude_mod=clf_config['exclude_modality'])
            model.load_state_dict(torch.load(os.path.join(m_path, config["path_classifier"])))
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
        else:
            model = setup_classifier(latent_dim=config['latent_dimension'], model=model, exclude_mod=config['exclude_modality'])

        model.to(device)

    if exclude_modality != config["exclude_modality"]:
        model.set_modalities(config["exclude_modality"])

    if latent_dim != config["latent_dimension"]:
        model.set_latent_dim(config["latent_dimension"])

    model.to(device)
    if config['adversarial_attack'] is not None:
        target_modality = config['target_modality']

        if config['adversarial_attack'] == 'gaussian_noise':
            attack = gaussian_noise.GaussianNoise(model=model, device=device, target_modality=target_modality, std=config['noise_std'])
        elif config['adversarial_attack'] == 'fgsm':
            attack = fgsm.FGSM(device=device, model=model, target_modality=target_modality, eps=config['adv_std'])
        elif config['adversarial_attack'] == 'pgd':
            attack = pgd.PGD(device=device, model=model, target_modality=target_modality, eps=config['adv_eps'], alpha=config['adv_alpha'], steps=config['adv_steps'])
        elif config['adversarial_attack'] == 'cw':
            attack = cw.CW(device=device, model=model, target_modality=target_modality, c_val=config['adv_c'], kappa=config['adv_kappa'], learning_rate=config['adv_lr'], steps=config['adv_steps'])

        if "classifier" in config['stage']:
            dataset.dataset = attack(dataset.dataset, dataset.labels)
        else:
            dataset.dataset = attack(dataset.dataset)

    if "notes" not in config:
        if 2 > 3:
            print('Enter experiment notes:')
            notes, _, _ = select.select([sys.stdin], [], [], TIMEOUT)
            if (notes):
                notes = sys.stdin.readline().strip()
            else:
                notes = ""
            #termios.tcflush(sys.stdin, termios.TCIOFLUSH)
        else:
            notes = None
    else:
        notes = config["notes"]

    if train:
        if config['optimizer'] is not None:
            if config['optimizer'] == 'adam':
                optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], betas=config['adam_betas'])
            elif config['optimizer'] == 'sgd':
                optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=config['momentum'])

        wandb.init(project="rmgm", 
               name=config['model_out'],
               config={key: value for key, value in config.items() if value is not None}, 
               notes=notes,
               allow_val_change=True,
               #magic=True,
               mode="offline",
               tags=[config['architecture'], config['dataset'], config['stage']])
        wandb.watch(model)
    else:
        optimizer = None

    for ckey, cval in config.items():
        if cval is not None:
            print(f'{ckey}: {cval}')
            with open(os.path.join(m_path, "results", config['stage'], config['model_out'] + '.txt'), 'a') as file:
                file.write(f'{ckey}: {cval}\n')

    return dataset, model, optimizer

@base_validation
def metrics_analysis(m_path, config):
    if "param_comp" not in config or config["param_comp"] is None:
        raise argparse.ArgumentError("Argument error: must define the hyperparameter to compare values.")
    if ("parent_param" in config and "parent_param" in ADVERSARIAL_ATTACKS) or "param_comp" in ADVERSARIAL_ATTACKS:
        if "target_modality" not in config or config["target_modality"] not in MODALITIES[config['dataset']]:
            raise argparse.ArgumentError(f"Argument error: must specify valid target modality to compare adversarial attacks.\n")
    if "model" in config['stage']:
        if config['architecture'] is None:
            raise argparse.ArgumentError("Argument error: must define a valid architecture when comparing metrics for train_model or test_model stages.")
        
        if config['architecture'] == 'vae':
            loss_dict = {'elbo_loss': [], 'kld_loss': []}
        elif config['architecture'] == 'dae':
            loss_dict = {'total_loss': []}
        elif config['architecture'] == 'gmc':
            loss_dict = {'infonce_loss': []}
        elif config['architecture'] == 'mvae':
            loss_dict = {'elbo_loss': [], 'kld_loss': []}
        elif config['architecture'] == 'dgmc':
            loss_dict = {'total_loss': [], 'infonce_loss': []}
        elif config['architecture'] == 'rgmc':
            loss_dict = {'total_loss': [], 'infonce_loss': [], 'o3n_loss': []}
        elif config['architecture'] == 'gmcwd':
            loss_dict = {'total_loss': [], 'infonce_loss': []}
        
        if config['dataset'] == 'mhd':
            if 'ae' in config['architecture'] or config['architecture'] == 'dgmc' or config['architecture'] == 'gmcwd':
                loss_dict = {**loss_dict, 'img_recon_loss': [], 'traj_recon_loss': []}
        elif config['dataset'] == 'mnist_svhn':
            if 'ae' in config['architecture'] or config['architecture'] == 'dgmc' or config['architecture'] == 'gmcwd':
                loss_dict = {**loss_dict, 'mnist_recon_loss': [], 'svhn_recon_loss': []}

    elif "classifier" in config['stage']:
        loss_dict = {'accuracy': []}
    else:
        raise ValueError(f"Invalid stage {config['stage']} for metric comparison.")

    try:
        os.makedirs(os.path.join(m_path, "compare", config['stage']), exist_ok=True)
    except IOError as e:
        traceback.print_exception(*sys.exc_info())
    finally:
        if "number_seeds" not in config or config["number_seeds"] is None or config["number_seeds"] == 0:
            if "train" in config['stage']:
                plot_loss_compare_graph(m_path, config, loss_dict)
            elif "test" in config['stage']:
                plot_metric_compare_bar(m_path, config, loss_dict)
        elif config["number_seeds"] > 1:
            if "train" in config['stage']:
                plot_bar_across_seeds(m_path, config, loss_dict)
            elif "test" in config['stage']:
                plot_graph_across_seeds(m_path, config, loss_dict)
        else:
            raise argparse.ArgumentError("Argument error: number_seeds parameter must be a non-negative integer.")
            
    return