import os
import json

from re import sub
from threading import Lock
from subprocess import check_output
from torch import optim, cuda, load

from utils.command_parser import create_idx_dict, config_validation
from input_transformations import gaussian_noise, fgsm, pgd, cw, bim

from architectures.mhd.models.vae import MhdVAE
from architectures.mhd.models.dae import MhdDAE
from architectures.mhd.models.mvae import MhdMVAE
from architectures.mhd.models.cmvae import MhdCMVAE
from architectures.mhd.models.cmdvae import MhdCMDVAE
from architectures.mhd.models.mdae import MhdMDAE
from architectures.mhd.models.cmdae import MhdCMDAE
from architectures.mhd.models.gmc import MhdGMC
from architectures.mhd.models.dgmc import MhdDGMC
from architectures.mhd.models.rgmc import MhdRGMC
from architectures.mhd.models.gmcwd import MhdGMCWD
from architectures.mhd.models.rgmcwd import MhdRGMCWD
from architectures.mhd.downstream.classifier import MHDClassifier
from architectures.mnist_svhn.models.vae import MSVAE
from architectures.mnist_svhn.models.dae import MSDAE
from architectures.mnist_svhn.models.mvae import MSMVAE
from architectures.mnist_svhn.models.cmvae import MSCMVAE
from architectures.mnist_svhn.models.cmdvae import MSCMDVAE
from architectures.mnist_svhn.models.mdae import MSMDAE
from architectures.mnist_svhn.models.cmdae import MSCMDAE
from architectures.mnist_svhn.models.gmc import MSGMC
from architectures.mnist_svhn.models.dgmc import MSDGMC
from architectures.mnist_svhn.models.rgmc import MSRGMC
from architectures.mnist_svhn.models.gmcwd import MSGMCWD
from architectures.mnist_svhn.downstream.classifier import MSClassifier
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


idx_lock = Lock()
device_lock = Lock()

def setup_device(m_path):
    if cuda.is_available():
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
        
        device_id = device_counter % cuda.device_count()
        device = f"cuda:{device_id}"
        device_lock.release()
    else:
        device = "cpu"

    return device

def setup_env(m_path, config):
    if cuda.is_available():
        config['device'] = cuda.get_device_name(cuda.current_device())
    else:
        command = "cat /proc/cpuinfo"
        all_info = check_output(command, shell=True).decode().strip()
        device_info = ""
        for line in all_info.split("\n"):
            if "model name" in line:
                device_info = sub( ".*model name.*:", "", line, 1)
                break

        config['device'] = device_info

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
        if "path_classifier" in config and config['path_classifier'] is not None:
            path = os.path.join(m_path, "configs", "train_classifier", config['path_classifier'] + ".json")
            clf_config = json.load(open(path))
            config['path_model'] = clf_config['path_model']
        else:
            config["path_model"] = config["architecture"] + "_" + config["dataset"] + f"_exp{exp_id}"

    if ("path_classifier" not in config or config["path_classifier"] is None) and config["stage"] == "test_classifier":
        config["path_classifier"] = "clf_" + config["path_model"]

    config = config_validation(m_path, config)

    return config

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

    dataset = setup_dataset(m_path, config, device, train)

    if "download" in config:
        config['download'] = None

    if config['stage'] != "train_model" and config['stage'] != "train_supervised":
        model_config = json.load(open(os.path.join(m_path, "configs", "train_model", config['path_model'] + '.json')))
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
        elif config['architecture'] == 'cmvae':
            scales = {'image': config['image_recon_scale'], 'trajectory': config['traj_recon_scale'], 'kld_beta': config['kld_beta']}
            model = MhdCMVAE(config['architecture'], latent_dim, device, exclude_modality, scales, config['rep_trick_mean'], config['rep_trick_std'])
        elif config['architecture'] == 'cmdvae':
            scales = {'image': config['image_recon_scale'], 'trajectory': config['traj_recon_scale'], 'kld_beta': config['kld_beta']}
            model = MhdCMDVAE(config['architecture'], latent_dim, device, exclude_modality, scales, config['rep_trick_mean'], config['rep_trick_std'], noise_factor=config['train_noise_factor'])
        elif config['architecture'] == 'mdae':
            scales = {'image': config['image_recon_scale'], 'trajectory': config['traj_recon_scale']}
            model = MhdMDAE(config['architecture'], latent_dim, device, exclude_modality, scales, noise_factor=config['train_noise_factor'])
        elif config['architecture'] == 'cmdae':
            scales = {'image': config['image_recon_scale'], 'trajectory': config['traj_recon_scale']}
            model = MhdCMDAE(config['architecture'], latent_dim, device, exclude_modality, scales, noise_factor=config['train_noise_factor'])
        elif config['architecture'] == 'gmc':
            model = MhdGMC(config['architecture'], exclude_modality, config['common_dimension'], latent_dim, config['infonce_temperature'])
        elif config['architecture'] == 'dgmc':
            scales = {'image': config['image_recon_scale'], 'trajectory': config['traj_recon_scale'], 'infonce_temp': config['infonce_temperature']}
            model = MhdDGMC(config['architecture'], exclude_modality, config['common_dimension'], latent_dim, scales, noise_factor=config['train_noise_factor'])
        elif config['architecture'] == 'rgmc':
            scales = {'infonce_temp': config['infonce_temperature'], 'o3n_loss_scale': config['o3n_loss_scale']}
            model = MhdRGMC(config['architecture'], exclude_modality, config['common_dimension'], latent_dim, scales, noise_factor=config['adv_std'], device=device)
        elif config['architecture'] == 'gmcwd':
            scales = {'image': config['image_recon_scale'], 'trajectory': config['traj_recon_scale'], 'infonce_temp': config['infonce_temperature']}
            model = MhdGMCWD(config['architecture'], exclude_modality, config['common_dimension'], latent_dim, scales, noise_factor=config['train_noise_factor'])
        elif config['architecture'] == 'rgmcwd':
            scales = {'image': config['image_recon_scale'], 'trajectory': config['traj_recon_scale'], 'infonce_temp': config['infonce_temperature'], 'o3n_loss_scale': config['o3n_loss_scale']}
            model = MhdRGMC(config['architecture'], exclude_modality, config['common_dimension'], latent_dim, scales, noise_factor=config['adv_std'], device=device)
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
        elif config['architecture'] == 'cmvae':
            scales = {'mnist': config['mnist_recon_scale'], 'svhn': config['svhn_recon_scale'], 'kld_beta': config['kld_beta']}
            model = MSCMVAE(config['architecture'], latent_dim, device, exclude_modality, scales, config['rep_trick_mean'], config['rep_trick_std'])
        elif config['architecture'] == 'cmvae':
            scales = {'mnist': config['mnist_recon_scale'], 'svhn': config['svhn_recon_scale'], 'kld_beta': config['kld_beta']}
            model = MSCMDVAE(config['architecture'], latent_dim, device, exclude_modality, scales, config['rep_trick_mean'], config['rep_trick_std'], noise_factor=config['train_noise_factor'])
        elif config['architecture'] == 'mdae':
            scales = {'mnist': config['mnist_recon_scale'], 'svhn': config['svhn_recon_scale']}
            model = MSMDAE(config['architecture'], latent_dim, device, exclude_modality, scales, noise_factor=config['train_noise_factor'])
        elif config['architecture'] == 'cmdae':
            scales = {'mnist': config['mnist_recon_scale'], 'svhn': config['svhn_recon_scale']}
            model = MSCMDAE(config['architecture'], latent_dim, device, exclude_modality, scales, noise_factor=config['train_noise_factor'])
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


    if config['stage'] == 'train_supervised':
        model = setup_classifier(latent_dim=config['latent_dimension'], model=model, exclude_mod=config['exclude_modality'])
    else:
        if "path_model" in config and config["path_model"] is not None and config["stage"] != "train_model":
            model.load_state_dict(load(os.path.join(m_path, "saved_models", config["path_model"] + ".pt")))
            model.eval()
            for param in model.parameters():
                param.requires_grad = False


        model.to(device)
        if 'classifier' in config['stage']:
            if "path_classifier" in config and config["path_classifier"] is not None and config["stage"] == "test_classifier":
                clf_config = json.load(open(os.path.join(m_path, "configs", "train_classifier", config['path_classifier'] + '.json')))
                model = setup_classifier(latent_dim=clf_config['latent_dimension'], model=model, exclude_mod=clf_config['exclude_modality'])
                model.load_state_dict(load(os.path.join(m_path, "saved_models", config["path_classifier"] + ".pt")))
                model.eval()
                for param in model.parameters():
                    param.requires_grad = False
            else:
                model = setup_classifier(latent_dim=config['latent_dimension'], model=model, exclude_mod=config['exclude_modality'])

            model.to(device)

        if exclude_modality != config["exclude_modality"]:
            model.set_modalities(config["exclude_modality"])
            model.to(device)

        if latent_dim != config["latent_dimension"]:
            model.set_latent_dim(config["latent_dimension"])
            model.to(device)

    if config['architecture'] == 'rgmc':
        if config['stage'] == 'train_model':
            gmc_config = json.load(open(os.path.join(m_path, "configs", "train_model", config['model_out'][1:] + '.json')))
        else:
            gmc_config = json.load(open(os.path.join(m_path, "configs", "train_model", config['model_out'][5:] + '.json')))
        if config['dataset'] == 'mhd':
            gmc_model = MhdGMC(gmc_config['architecture'], gmc_config['exclude_modality'], gmc_config['common_dimension'], gmc_config['latent_dimension'], gmc_config['infonce_temperature'])
        elif config['dataset'] == 'mnist_svhn':
            gmc_model = MSGMC(gmc_config['architecture'], gmc_config['exclude_modality'], gmc_config['common_dimension'], gmc_config['latent_dimension'], gmc_config['infonce_temperature'])
        
        gmc_model.load_state_dict(load(os.path.join(m_path, "saved_models", gmc_config["model_out"] + ".pt")))
        for param in gmc_model.parameters():
            param.requires_grad = False

        gmc_model.to(device)
        if config['stage'] == 'train_model':
            clf_gmc_config = json.load(open(os.path.join(m_path, "configs", "train_classifier", 'clf_' + config['model_out'][1:] + '.json')))
        else:
            clf_gmc_config = json.load(open(os.path.join(m_path, "configs", "train_classifier", 'clf_' + config['model_out'][5:] + '.json')))
        clf_gmc_model = setup_classifier(latent_dim=gmc_config['latent_dimension'], model=gmc_model, exclude_mod=gmc_config['exclude_modality'])
        clf_gmc_model.load_state_dict(load(os.path.join(m_path, "saved_models", clf_gmc_config["model_out"] + ".pt")))
        clf_gmc_model.eval()
        for param in clf_gmc_model.parameters():
            param.requires_grad = False

        clf_gmc_model.to(device)
        attack = fgsm.FGSM(device=device, model=clf_gmc_model, target_modality=None, eps=config['adv_std'])
        model.set_perturbation(attack)

    if config['adversarial_attack'] is not None:
        target_modality = config['target_modality']

        if config['adversarial_attack'] == 'gaussian_noise':
            attack = gaussian_noise.GaussianNoise(device=device, target_modality=target_modality, std=config['noise_std'])
        elif config['adversarial_attack'] == 'fgsm':
            attack = fgsm.FGSM(device=device, model=model, target_modality=target_modality, eps=config['adv_std'])
        elif config['adversarial_attack'] == 'bim':
            attack = bim.BIM(device=device, model=model, target_modality=target_modality, eps=config['adv_std'], alpha=config['adv_alpha'], steps=config['adv_steps'])
        elif config['adversarial_attack'] == 'pgd':
            attack = pgd.PGD(device=device, model=model, target_modality=target_modality, eps=config['adv_eps'], alpha=config['adv_alpha'], steps=config['adv_steps'])
        elif config['adversarial_attack'] == 'cw':
            attack = cw.CW(device=device, model=model, target_modality=target_modality, c_val=config['adv_eps'], kappa=config['adv_kappa'], learning_rate=config['adv_lr'], steps=config['adv_steps'])

        if "classifier" in config['stage'] or config['stage'] == 'train_supervised':
            dataset.dataset = attack(dataset.dataset, dataset.labels)
        else:
            dataset.dataset = attack(dataset.dataset)

    if False:
        if "notes" not in config:
            print('Enter experiment notes:')
            notes, _, _ = select.select([sys.stdin], [], [], TIMEOUT)
            if (notes):
                notes = sys.stdin.readline().strip()
            else:
                notes = ""
            #termios.tcflush(sys.stdin, termios.TCIOFLUSH)
        else:
            notes = config["notes"]

    if train and config['stage'] != 'inference':
        if config['optimizer'] is not None:
            if config['optimizer'] == 'adam':
                optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], betas=config['adam_betas'])
            elif config['optimizer'] == 'sgd':
                optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=config['momentum'])

        if False:
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
        
    return dataset, model, optimizer