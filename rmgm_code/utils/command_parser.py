import re
import os
import sys
import json
import torch
import wandb
import select
import shutil
import termios
import argparse
import threading
import itertools
import traceback
import subprocess

import torch.optim as optim

from torchvision import transforms
from input_transformations import gaussian_noise, fgsm
from architectures.downstream import classifier
from architectures.models import vae, dae, gmc, mvae, dgmc
from datasets.mhd.MHDDataset import MHDDataset
from datasets.mosi.MOSIDataset import MOSIDataset
from datasets.mosei.MOSEIDataset import MOSEIDataset
from datasets.pendulum.PendulumDataset import PendulumDataset

TIMEOUT = 0 # Seconds to wait for user to input notes
ARCHITECTURES = ['vae', 'dae', 'gmc', 'mvae', 'dgmc']
DATASETS = ['mhd', 'mosi', 'mosei', 'pendulum']
OPTIMIZERS = ['sgd', 'adam', None]
NOISE_TYPES = ['gaussian', None] 
ADVERSARIAL_ATTACKS = ["fgsm", None]
EXPERTS_FUSION_TYPES = ['poe', 'moe', None]
MODALITIES = ['image', 'trajectory', None]
STAGES = ['train_model', 'train_classifier', 'test_model', 'test_classifier', 'inference']

SEED = 42
LR_DEFAULT = 0.001
EPOCHS_DEFAULT = 15
BATCH_SIZE_DEFAULT = 256
CHECKPOINT_DEFAULT = 0
LATENT_DIM_DEFAULT = 256
INFONCE_TEMPERATURE_DEFAULT = 0.2
RECON_SCALE_DEFAULTS = {'image': 0.5, 'trajectory': 0.5}
KLD_BETA_DEFAULT = 0.5
REPARAMETERIZATION_MEAN_DEFAULT = 0.0
REPARAMETERIZATION_STD_DEFAULT = 1.0
EXPERTS_FUSION_DEFAULT = "poe"
POE_EPS_DEFAULT = 1e-8
MOMENTUM_DEFAULT = 0.9
ADAM_BETAS_DEFAULTS = [0.9, 0.999]
NOISE_STD_DEFAULT = 1.0
ADV_EPSILON_DEFAULT = 8 / 255

idx_lock = threading.Lock()
device_lock = threading.Lock()

def process_arguments(m_path):
    parser = argparse.ArgumentParser(prog="rmgm", description="Program tests the performance and robustness of several generative models with clean and noisy/adversarial samples.")
    subparsers = parser.add_subparsers(help="command", dest="command")
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


def config_validation(m_path, config):
    if "stage" not in config or config["stage"] not in STAGES:
        raise argparse.ArgumentError("Argument error: must specify a valid pipeline stage.")
    if "architecture" not in config or config["architecture"] not in ARCHITECTURES:
        raise argparse.ArgumentError("Argument error: must specify an architecture for the experiments.")
    if "dataset" not in config or config["dataset"] not in DATASETS:
        raise argparse.ArgumentError("Argument error: must specify a dataset for the experiments.")
    
    try:
        os.makedirs(os.path.join(m_path, "results", config['stage']), exist_ok=True)
    except IOError as e:
        traceback.print_exception(*sys.exc_info())
    finally:
        if config['stage'] == 'train_model' and config['model_out'] is None:
            raise argparse.ArgumentError('Argument error: the --model_out argument must be set when the --stage argument is ' + config['stage'] + '.')
        if config['batch_size'] < 1:
            raise argparse.ArgumentError("Argument error: batch_size value must be a positive and non-zero integer.")
        
        if config['stage'] == 'train_model' or config['stage'] == 'train_classifier':
            os.makedirs(os.path.join(m_path, "configs", config['stage']), exist_ok=True)
            if config['checkpoint'] < 0:
                raise argparse.ArgumentError("Argument error: checkpoint value must be an integer greater than or equal to 0.")
            elif config['checkpoint'] > config['epochs']:
                raise argparse.ArgumentError("Argument error: checkpoint value must be smaller than or equal to the number of epochs.")
            if "epochs" not in config or config["epochs"] is None:
                config['epochs'] = EPOCHS_DEFAULT
        else:
            if "epochs" in config and config["epochs"] is not None:
                config["epochs"] = None
            if "learning_rate" in config and config['learning_rate'] is not None:
                config['learning_rate'] = None
            if "optimizer" in config and config['optimizer'] is not None:
                config['optimizer'] = None
            if config["stage"] != "inference" and "checkpoint" in config and config["checkpoint"] is not None:
                config["checkpoint"] = None

        if "seed" not in config or config['seed'] is None:
            config["seed"] = SEED
    
        if "latent_dimension" not in config or config['latent_dimension'] is None:
            config['latent_dimension'] = LATENT_DIM_DEFAULT

        if config['latent_dimension'] < 1:
            raise argparse.ArgumentError("Argument error: latent_dimension value must be a positive and non-zero integer.")

        if "exclude_modality" not in config:
            config["exclude_modality"] = None

        if "target_modality" not in config:
            config['target_modality'] = None

        if config['exclude_modality'] is not None and config['target_modality'] is not None and config['exclude_modality'] == config['target_modality']:
            raise argparse.ArgumentError("Argument error: target modality cannot be the same as excluded modality.")

        if "download" not in config:
            config["download"] = False

        if "batch_size" not in config or config["batch_size"] is None:
            config['batch_size'] = BATCH_SIZE_DEFAULT


        if "ae" in config['architecture'] or config['architecture'] == "dgmc":
            if "image_recon_scale" not in config or config['image_recon_scale'] is None:
                config["image_recon_scale"] = RECON_SCALE_DEFAULTS['image']
            if "traj_recon_scale" not in config or config['traj_recon_scale'] is None:
                config["traj_recon_scale"] = RECON_SCALE_DEFAULTS['trajectory']

            if "vae" in config['architecture']:
                if "kld_beta" not in config or config['kld_beta'] is None:
                    config['kld_beta'] = KLD_BETA_DEFAULT
                if "rep_trick_mean" not in config:
                    config['rep_trick_mean'] = REPARAMETERIZATION_MEAN_DEFAULT
                if "rep_trick_std" not in config:
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

            if config['exclude_modality'] == 'image':
                config['image_recon_scale'] = 0.
                if config['target_modality'] == 'image':
                    config['target_modality'] = None
            elif config['exclude_modality'] == 'trajectory':
                config['traj_recon_scale'] = 0.
                if config['target_modality'] == 'trajectory':
                    config['target_modality'] = None
        else:
            config['image_recon_scale'] = None
            config['traj_recon_scale'] = None
            config['kld_beta'] = None
            config['rep_trick_mean'] = None
            config['rep_trick_std'] = None
            config['experts_fusion'] = None
            config['poe_eps'] = None
            

        if config['stage'] == "train_model":
            config["path_model"] = None

        if "model" in config['stage']:
            config["path_classifier"] = None
        
        if config['stage'] == 'test_classifier':
            if "path_classifier" not in config or config['path_classifier'] is None:
                config['path_classifier'] = os.path.join(os.path.dirname(config['path_model']), "clf_" + os.path.basename(config['path_model']))

        if "gmc" in config['architecture']:
            if "infonce_temperature" not in config or config['infonce_temperature'] is None:
                config["infonce_temperature"] = INFONCE_TEMPERATURE_DEFAULT
        else:
            config['infonce_temperature'] = None

        if "noise" not in config:
            config["noise"] = None

        if config['noise'] is None:
            config["noise_std"] = None
        else:
            if "noise_std" not in config or config["noise_std"] is None:
                config["noise_std"] = NOISE_STD_DEFAULT

        if "adversarial_attack" not in config:
            config["adversarial_attack"] = None

        if config["adversarial_attack"] is None:
            config['adv_epsilon'] = None
        elif 'adv_epsilon' not in config or config['adv_epsilon'] is None:
            config['adv_epsilon'] = ADV_EPSILON_DEFAULT

        if config["noise"] is not None or config["adversarial_attack"] is not None:
            if config["target_modality"] is None:
                raise argparse.ArgumentError("Argument error: must define the target_modality for noise/adversarial attack.")

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
                device_info = re.sub( ".*model name.*:", "", line,1)
                break

        config['device'] = device_info

    return device


def setup_experiment(m_path, config, device, train=True):
    def setup_dataset(m_path, config, device, train):
        if config['dataset'] == 'mhd':
            dataset = MHDDataset(os.path.join(m_path, "datasets", "mhd"), device, config['download'], config['exclude_modality'], config['target_modality'], train)
        elif config['dataset'] == 'mosi':
            dataset = MOSIDataset(os.path.join(m_path, "datasets", "mosi"), device, config['download'], config['exclude_modality'], config['target_modality'], train)
        elif config['dataset'] == 'mosei':
            dataset = MOSEIDataset(os.path.join(m_path, "datasets", "mosei"), device, config['download'], config['exclude_modality'], config['target_modality'], train)
        elif config['dataset'] == 'pendulum':
            dataset = PendulumDataset(os.path.join(m_path, "datasets", "pendulum"), device, config['download'], config['exclude_modality'], config['target_modality'], train)
        return dataset

    if config['stage'] == "inference":
        dataset = setup_dataset(m_path, config, device, True)
        test_dataset = setup_dataset(m_path, config, device, False)
        dataset = torch.concat((dataset, test_dataset), dim=0)
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

    if config['architecture'] == 'vae':
        scales = {'image': config['image_recon_scale'], 'trajectory': config['traj_recon_scale'], 'kld_beta': config['kld_beta']}
        model = vae.VAE(config['architecture'], latent_dim, device, exclude_modality, scales, config['rep_trick_mean'], config['rep_trick_std'])
    elif config['architecture'] == 'dae':
        scales = {'image': config['image_recon_scale'], 'trajectory': config['traj_recon_scale']}
        model = dae.DAE(config['architecture'], latent_dim, device, exclude_modality, scales)
    elif config['architecture'] == 'gmc':
        model = gmc.MhdGMC(config['architecture'], exclude_modality, latent_dim, config['infonce_temperature'])
    elif config['architecture'] == 'mvae':
        scales = {'image': config['image_recon_scale'], 'trajectory': config['traj_recon_scale'], 'kld_beta': config['kld_beta']}
        model = mvae.MVAE(config['architecture'], latent_dim, device, exclude_modality, scales, config['rep_trick_mean'], config['rep_trick_std'], config['experts_fusion'], config['poe_eps'])
    elif config['architecture'] == 'dgmc':
        scales = {'image': config['image_recon_scale'], 'trajectory': config['traj_recon_scale'], 'infonce_temp': config['infonce_temperature']}
        model = dgmc.MhdDGMC(config['architecture'], exclude_modality,latent_dim, scales)

    if "path_model" in config and config["path_model"] is not None and config["stage"] != "train_model":
        model.load_state_dict(torch.load(os.path.join(m_path, config["path_model"])))
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    model.to(device)

    if 'classifier' in config['stage']:
        if "path_classifier" in config and config["path_classifier"] is not None and config["stage"] == "test_classifier":
            clf_config = json.load(open(os.path.join(m_path, "configs", "train_classifier", os.path.basename(os.path.splitext(config['path_classifier'])[0]) + '.json')))
            model = classifier.MNISTClassifier(clf_config['latent_dimension'], model, clf_config['exclude_modality'])
            model.load_state_dict(torch.load(os.path.join(m_path, config["path_classifier"])))
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
        else:
            model = classifier.MNISTClassifier(config['latent_dimension'], model, config['exclude_modality'])
        model.to(device)

    if exclude_modality != config["exclude_modality"]:
        model.set_modalities(config["exclude_modality"])

    if latent_dim != config["latent_dimension"]:
        model.set_latent_dim(config["latent_dimension"])

    model.to(device)

    if config['adversarial_attack'] is not None or config['noise'] is not None:
        target_modality = config['target_modality']
        noise = config['noise']
        
        if noise == "gaussian":
            transform = transforms.Compose([gaussian_noise.GaussianNoise(device, target_modality, 0. ,config['noise_std'])])
        else:
            transform = None

        dataset._set_transform(transform)

        if config['adversarial_attack'] == 'fgsm':
            adv_attack = fgsm.FGSM(device, model, target_modality, eps=config['adv_epsilon'])
        else:
            adv_attack = None

        dataset._set_adv_attack(adv_attack)

    if "notes" not in config:
        print('Enter experiment notes:')
        notes, _, _ = select.select([sys.stdin], [], [], TIMEOUT)
        if (notes):
            notes = sys.stdin.readline().strip()
        else:
            notes = ""
        termios.tcflush(sys.stdin, termios.TCIOFLUSH)
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