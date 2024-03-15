import os
import sys
import wandb
import traceback

from torch import device as torch_device

from time import sleep
from utils.train import run_training
from utils.test import run_test, run_inference
from utils.command_parser import process_arguments
from utils.setup import setup_experiment, setup_env, setup_device

# Assign path to current directory
m_path = os.getcwd()

WAIT_TIME = 0 # Seconds to wait between sequential experiments

def train_model(config, device):
    dataset, model, optimizer = setup_experiment(m_path, config, device, train=True)
    model = run_training(m_path, config, device, dataset, model, optimizer)
    return model

def train_downstream_classifier(config, device):
    dataset, model, optimizer = setup_experiment(m_path, config, device, train=True)
    model = run_training(m_path, config, device, dataset, model, optimizer)
    return model

def train_supervised_model(config, device):
    dataset, model, optimizer = setup_experiment(m_path, config, device, train=True)
    model = run_training(m_path, config, device, dataset, model, optimizer)
    return model

def test_model(config, device):
    dataset, model, _ = setup_experiment(m_path, config, device, train=False)
    run_test(m_path, config, device, model, dataset)
    return

def test_downstream_classifier(config, device):
    dataset, model, _ = setup_experiment(m_path, config, device, train=False)
    run_test(m_path, config, device, model, dataset)
    return

def inference(config, device):
    dataset, model, _ = setup_experiment(m_path, config, device, train=True)
    run_inference(m_path, config, device, model, dataset)
    return

def call_with_configs(config_ls):
    def decorate(run_experiment):
        def wrapper(*args, **kwargs):
            device = setup_device(m_path)
            for config in config_ls:
                config = setup_env(m_path, config)
                kwargs['device'] = torch_device(device)
                kwargs['config'] = config
                print(f'Starting up experiment on device {device}...')
                run_experiment(**kwargs)
                print(f'Finishing up experiment on device {device}...')
                sleep(WAIT_TIME)
        return wrapper
    return decorate

def run_experiment(**kwargs):
    config = kwargs['config']
    device = kwargs['device']
    try:
        if config['stage'] == 'train_model':
            train_model(config, device)
        elif config['stage'] == 'train_classifier':
            train_downstream_classifier(config, device)
        elif config['stage'] == 'train_supervised':
            train_supervised_model(config, device)
        elif config['stage'] == 'test_model':
            test_model(config, device)
        elif config['stage'] == 'test_classifier':
            test_downstream_classifier(config, device)
        elif config['stage'] == 'inference':
            inference(config, device)
    except:
        if 'wandb' in config and config['wandb']:
            wandb.finish(exit_code=1)
        traceback.print_exception(*sys.exc_info())

def main():
    try:
        os.makedirs(os.path.join(m_path, "results"), exist_ok=True)
        os.makedirs(os.path.join(m_path, "compare"), exist_ok=True)
        os.makedirs(os.path.join(m_path, "configs"), exist_ok=True)
        os.makedirs(os.path.join(m_path, "saved_models"), exist_ok=True)
        os.makedirs(os.path.join(m_path, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(m_path, "tmp"), exist_ok=True)
    except IOError as e:
        traceback.print_exception(*sys.exc_info())
    finally:
        configs = process_arguments(m_path)
        call_with_configs(config_ls=configs)(run_experiment)()

if __name__ == "__main__":
    main()
    sys.exit(0)