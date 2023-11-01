import os
import sys
import json
import shutil
import random
import argparse
import traceback
import numpy as np

from itertools import product
from torch.backends import cudnn
from torch import manual_seed, cuda
from utils.logger import plot_loss_compare_graph, plot_metric_compare_bar, plot_bar_across_models, save_config


TIMEOUT = 0 # Seconds to wait for user to input notes
ARCHITECTURES = ['vae', 'dae', 'mvae', 'cmvae', 'cmdvae', 'mdae', 'cmdae', 'gmc', 'dgmc', 'gmcwd', 'rgmc']
DATASETS = ['mhd', 'mnist_svhn', 'mosi', 'mosei', 'pendulum']
OPTIMIZERS = ['sgd', 'adam', None]
ADVERSARIAL_ATTACKS = ["gaussian_noise", "fgsm", "pgd", "bim", None]
EXPERTS_FUSION_TYPES = ['poe', 'moe', None]
STAGES = ['train_model', 'train_classifier', 'test_model', 'test_classifier', 'inference']
MODALITIES = {
    'mhd': ['image', 'trajectory'],
    'mnist_svhn': ['mnist', 'svhn'],
    'pendulum': ['image', 'sound']
}

SEED = 42
LR_DEFAULT = 0.001
EPOCHS_DEFAULT = 100
BATCH_SIZE_DEFAULT = 64
CHECKPOINT_DEFAULT = 0
LATENT_DIM_DEFAULT = 64
COMMON_DIM_DEFAULT = 64
INFONCE_TEMPERATURE_DEFAULT = 0.1
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
ADV_EPSILON_DEFAULT = 0.15
ADV_ALPHA_DEFAULT = 25 / 255
ADV_STEPS_DEFAULT = 10
ADV_KAPPA_DEFAULT = 10
ADV_LR_DEFAULT = 0.001
RECON_SCALE_DEFAULTS = {
    'mhd': {'image': 0.5, 'trajectory': 0.5}, 
    'mnist_svhn': {'mnist': 0.5, 'svhn': 0.5}
}

def process_arguments(m_path):
    parser = argparse.ArgumentParser(prog="rmgm", description="Program tests the performance and robustness of several different models with clean and noisy/adversarial samples.")
    subparsers = parser.add_subparsers(help="command", dest="command")
    comp_parser = subparsers.add_parser("compare")
    comp_parser.add_argument('-a', '--architecture', choices=ARCHITECTURES + ["all"], help='Architecture to be used in the comparison.')
    comp_parser.add_argument('-d', '--dataset', type=str, default='mhd', choices=DATASETS, help='Dataset to be used in the comparison.')
    comp_parser.add_argument('-s', '--stage', type=str, default='train_model', choices=STAGES, help='Stage of the pipeline to be used in the comparison.')
    comp_parser.add_argument('--model_outs', '--mos', type=int, nargs='+')
    comp_parser.add_argument('--compare_models', '--cm', action="store_true")
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
    clear_parser.add_argument('--clear_compare', '--clear_comp', action="store_false", help="Flag to delete compare directory.")
    clear_parser.add_argument('--clear_saved_models', '--clear_models', '--clear_saved', action="store_false", help="Flag to delete saved_models directory.")
    clear_parser.add_argument('--clear_configs', '--clear_runs', action="store_false", help="Flag to delete configs directory.")
    clear_parser.add_argument('--clear_wandb', '--clear_wb', action="store_false", help="Flag to delete wandb directory.")
    clear_parser.add_argument('--clear_idx', action="store_false", help="Flag to delete previous experiments idx file.")

    configs_parser = subparsers.add_parser("config")
    configs_parser.add_argument('--load_config', '--load_json', type=str, help='File path where the experiment(s) configurations are to loaded from.')
    configs_parser.add_argument('--config_permute', '--permute_config', type=str, nargs='+', help='Generate several config runs from permutations of dict of lists with hyperparams.')
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
    exp_parser.add_argument('--image_recon_scale', type=float, default=RECON_SCALE_DEFAULTS['mhd']['image'], help='Weight for the image reconstruction loss.')
    exp_parser.add_argument('--traj_recon_scale', type=float, default=RECON_SCALE_DEFAULTS['mhd']['trajectory'], help='Weight for the trajectory reconstruction loss.')
    exp_parser.add_argument('--mnist_recon_scale', type=float, default=RECON_SCALE_DEFAULTS['mnist_svhn']['mnist'], help='Weight for the mnist reconstruction loss.')
    exp_parser.add_argument('--svhn_recon_scale', type=float, default=RECON_SCALE_DEFAULTS['mnist_svhn']['svhn'], help='Weight for the svhn reconstruction loss.')
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
    exp_parser.add_argument('--black_box', action="store_true", help='Defines in adversarial attack is performed in a black-box setting.')
    exp_parser.add_argument('--download', type=bool, default=False, help='If true, downloads the choosen dataset.')
    
    args = vars(parser.parse_args())
    if args['command'] == 'compare':
        config = {
            'architecture': args['architecture'],
            'dataset': args['dataset'],
            'stage': args['stage'],
            'model_outs': args['model_outs'],
            'param_comp': args['param_comp'],
            'parent_param': args['parent_param'],
            'number_seeds': args['number_seeds'],
            'compare_models': args['compare_models'],
            'target_modality': args['target_modality']
        }

        if config['architecture'] == 'all':
            for architecture in ARCHITECTURES:
                config['architecture'] = architecture
                metrics_analysis(m_path, config)
        else:
            metrics_analysis(m_path, config)
        sys.exit(0)

    if args['command'] == 'clear':
        if args['clear_results']:
            shutil.rmtree(os.path.join(m_path, "results"), ignore_errors=True)
        if args['clear_checkpoints']:
            shutil.rmtree(os.path.join(m_path, "checkpoints"), ignore_errors=True)
        if args['clear_compare']:
            shutil.rmtree(os.path.join(m_path, "compare"), ignore_errors=True)
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

    if args['command'] == 'config':
        if "config_permute" in args and args['config_permute'] is not None:
            configs = []
            for partial_path in args['config_permute']:
                conf_path = open(os.path.join(m_path, partial_path))
                hyperparams = json.load(conf_path)
                keys, values = zip(*hyperparams.items())
                configs.append([dict(zip(keys, v)) for v in product(*values)])
            configs = [x for subconf in configs for x in subconf]
        else:
            configs = []
            for partial_path in args['load_config']:
                configs.append(open(os.path.join(m_path, args['load_config'])))
        return configs
    
    if args['command'] == 'experiment':
        args.pop('command')
        return [args]

    raise argparse.ArgumentError("Argumment error: unknown command " + args['command'])

def base_validation(function):
    def wrapper(m_path, config):
        if "stage" not in config or config["stage"] not in STAGES:
            raise argparse.ArgumentError("Argument error: must specify a valid pipeline stage.")
        if "dataset" not in config or config["dataset"] not in DATASETS:
            raise argparse.ArgumentError("Argument error: must specify a dataset for the experiments.")
        return function(m_path, config)
    return wrapper

@base_validation
def config_validation(m_path, config):
    if "architecture" not in config or config["architecture"] not in ARCHITECTURES:
            raise argparse.ArgumentError("Argument error: must define a valid architecture.")
    if "batch_size" not in config or config["batch_size"] is None:
            config['batch_size'] = BATCH_SIZE_DEFAULT
    if config['batch_size'] < 1:
        raise argparse.ArgumentError("Argument error: batch_size value must be a positive and non-zero integer.")
    if "latent_dimension" not in config or config['latent_dimension'] is None:
        config['latent_dimension'] = LATENT_DIM_DEFAULT
    if config['latent_dimension'] < 1:
        raise argparse.ArgumentError("Argument error: latent_dimension value must be a positive and non-zero integer.")
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
    
    if config['stage'] == 'test_classifier':
        if "path_classifier" in config and config['path_classifier'] is not None:
            path = os.path.join(m_path, "configs", "train_classifier", config['path_classifier'] + ".json")
            clf_config = json.load(open(path))
        else:
            clf_config = None

        if "path_model" in config and config['path_model'] is not None:
            path = os.path.join(m_path, "configs", "train_model", config['path_model'] + ".json")
            model_config = json.load(open(path))
        else:
            model_config = None
    elif config['stage'] == 'train_classifier' or config['stage'] == 'test_model':
        if "path_model" in config and config['path_model'] is not None:
            path = os.path.join(m_path, "configs", "train_model", config['path_model'] + ".json")
            model_config = json.load(open(path))
        else:
            model_config = None
        clf_config = None
    else:
        model_config = None
        clf_config = None

    if "seed" not in config or config['seed'] is None:
        if clf_config is not None:
            config['seed'] = clf_config['seed']
        elif model_config is not None:
            config['seed'] = model_config['seed']
        else:
            config["seed"] = SEED

    seed = config['seed']
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    manual_seed(seed)
    cuda.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True
    
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
        if "exclude_modality" not in config:
            config["exclude_modality"] = None

        if "target_modality" not in config:
            config['target_modality'] = None

        if "download" not in config:
            config["download"] = None

        if "ae" in config['architecture'] or config['architecture'] == "dgmc" or config['architecture'] == 'gmcwd':
            if config['dataset'] == "mhd":
                if "image_recon_scale" not in config:
                    config["image_recon_scale"] = RECON_SCALE_DEFAULTS['mhd']['image']
                if "traj_recon_scale" not in config:
                    config["traj_recon_scale"] = RECON_SCALE_DEFAULTS['mhd']['trajectory']
            elif config['dataset'] == "mnist_svhn":
                if "mnist_recon_scale" not in config:
                    config["mnist_recon_scale"] = RECON_SCALE_DEFAULTS['mnist_svhn']['mnist']
                if "svhn_recon_scale" not in config:
                    config["svhn_recon_scale"] = RECON_SCALE_DEFAULTS['mnist_svhn']['svhn']

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

                if "ae" in config['architecture'] or config['architecture'] == "dgmc" or config['architecture'] == 'gmcwd':
                    if "mnist_recon_scale" in config and config["mnist_recon_scale"] is not None:
                        config["mnist_recon_scale"] = None
                    if "svhn_recon_scale" in config and config["svhn_recon_scale"] is not None:
                        config["svhn_recon_scale"] = None
            elif config['dataset'] == 'mnist_svhn':
                if config['exclude_modality'] == 'mnist':
                    config['mnist_recon_scale'] = 0.
                elif config['exclude_modality'] == 'svhn':
                    config['svhn_recon_scale'] = 0.

                if "ae" in config['architecture'] or config['architecture'] == "dgmc" or config['architecture'] == 'gmcwd':
                    if "image_recon_scale" in config and config["image_recon_scale"] is not None:
                        config["image_recon_scale"] = None
                    if "traj_recon_scale" in config and config["traj_recon_scale"] is not None:
                        config["traj_recon_scale"] = None
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
                else:
                    config['adv_epsilon'] = None
                    config['adv_alpha'] = None
                    config['adv_steps'] = None
                    config['adv_kappa'] = None
                    config['adv_lr'] = None
            else:
                config["noise_std"] = None
                if 'adv_epsilon' not in config or config['adv_epsilon'] is None:
                    config['adv_epsilon'] = ADV_EPSILON_DEFAULT
                    
                if config["adversarial_attack"] == 'pgd' or config["adversarial_attack"] == 'cw' or config["adversarial_attack"] == 'bim':
                    if 'adv_steps' not in config or config['adv_steps'] is None:
                            config['adv_steps'] = ADV_STEPS_DEFAULT
                    
                    if config['adversarial_attack'] == 'cw':
                        if 'adv_kappa' not in config or config['adv_kappa'] is None:
                            config['adv_kappa'] = ADV_KAPPA_DEFAULT
                        if 'adv_lr' not in config or config['adv_lr'] is None:
                            config['adv_lr'] = ADV_LR_DEFAULT
                    else:
                        config['adv_kappa'] = None
                        config['adv_lr'] = None

                    if config["adversarial_attack"] == 'pgd' or config["adversarial_attack"] == 'bim':
                        if 'adv_alpha' not in config or config['adv_alpha'] is None:
                            config['adv_alpha'] = ADV_ALPHA_DEFAULT
                    else:
                        config['adv_alpha'] = None
                else:
                    config['adv_alpha'] = None
                    config['adv_steps'] = None
                    config['adv_kappa'] = None
                    config['adv_lr'] = None

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

    save_config(os.path.join(m_path, "results", config['stage'], config['model_out'] + '.txt'), config)
    return config

def create_idx_dict():
    idx_dict = {}
    for stage in STAGES:
        idx_dict[stage] = {}
        for dataset in DATASETS:
            idx_dict[stage][dataset] = dict.fromkeys(ARCHITECTURES, 0)
    return idx_dict

@base_validation
def metrics_analysis(m_path, config):
    if ("parent_param" in config and "parent_param" in ADVERSARIAL_ATTACKS) or "param_comp" in ADVERSARIAL_ATTACKS:
        if "target_modality" not in config or config["target_modality"] not in MODALITIES[config['dataset']]:
            raise argparse.ArgumentError(f"Argument error: must specify valid target modality to compare adversarial attacks.\n")
    if 'model_outs' not in config or config['model_outs'] is None:
        raise argparse.ArgumentError("Argument error: model_outs must contain at least one positive integer.")
    else:
        for mos in config['model_outs']:
            if mos <= 0:
                raise argparse.ArgumentError("Argument error: all model_outs must be positive integer.")
    if 'number_seeds' not in config or config['number_seeds'] is None or config['number_seeds'] <= 0:
        raise argparse.ArgumentError("Argument error: number_seeds parameter must be a positive integer.")
    if "model" in config['stage']:
        if config['architecture'] is None:
            raise argparse.ArgumentError("Argument error: must define a valid architecture when comparing metrics for train_model or test_model stages.")
        
        if config['architecture'] == 'vae':
            loss_dict = {'elbo_loss': []}
        elif config['architecture'] == 'dae':
            loss_dict = {'total_loss': []}
        elif config['architecture'] == 'gmc':
            loss_dict = {'infonce_loss': []}
        elif config['architecture'] == 'mvae':
            loss_dict = {'elbo_loss': []}
        elif config['architecture'] == 'cmvae':
            loss_dict = {'total_loss': []}
        elif config['architecture'] == 'cmdvae':
            loss_dict = {'total_loss': []}
        elif config['architecture'] == 'mdae':
            loss_dict = {'total_loss': []}
        elif config['architecture'] == 'cmdae':
            loss_dict = {'total_loss': []}
        elif config['architecture'] == 'dgmc':
            loss_dict = {'total_loss': []}
        elif config['architecture'] == 'rgmc':
            loss_dict = {'total_loss': []}
        elif config['architecture'] == 'gmcwd':
            loss_dict = {'total_loss': []}
        

    elif "classifier" in config['stage'] or "train_supervised" in config['stage']:
        loss_dict = {'accuracy': []}
    else:
        raise ValueError(f"Invalid stage {config['stage']} for metric comparison.")

    try:
        os.makedirs(os.path.join(m_path, "compare", config['stage']), exist_ok=True)
    except IOError as e:
        traceback.print_exception(*sys.exc_info())
    finally:
        last_id = int(config['model_outs'][-1]) + int(config['number_seeds'])
        out_path = f"{config['dataset']}_{config['model_outs'][0]}_{last_id}_"
        if config['compare_models']:
            plot_bar_across_models(m_path, config, out_path, ARCHITECTURES)
        else:
            out_path = f"{config['architecture']}_{out_path}"
            save_config(os.path.join(m_path, "compare", config['stage'], out_path + 'metrics.txt'), config)
            if "train" in config['stage']:
                plot_loss_compare_graph(m_path, config, loss_dict, out_path)
            elif "test" in config['stage']:
                plot_metric_compare_bar(m_path, config, loss_dict, out_path)
            else:
                raise argparse.ArgumentError("Argument error: cannot compare results for inference stage.")
            
    return