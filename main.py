import os
import sys
import time
import torch
import wandb
import traceback

from utils.definitions import ROOT_DIR, WAIT_TIME
from utils.train import run_training
from utils.test import run_test, run_inference
from utils.command_parser import process_arguments
from utils.setup import setup_experiment, setup_env, setup_device


def train_model(config, device):
    dataset, model, optimizer = setup_experiment(ROOT_DIR, config, device, train=True)
    model = run_training(ROOT_DIR, config, device, dataset, model, optimizer)
    return model


def train_downstream_classifier(config, device):
    dataset, model, optimizer = setup_experiment(ROOT_DIR, config, device, train=True)
    model = run_training(ROOT_DIR, config, device, dataset, model, optimizer)
    return model


def train_supervised_model(config, device):
    dataset, model, optimizer = setup_experiment(ROOT_DIR, config, device, train=True)
    model = run_training(ROOT_DIR, config, device, dataset, model, optimizer)
    return model


def test_model(config, device):
    dataset, model, _ = setup_experiment(ROOT_DIR, config, device, train=False)
    run_test(ROOT_DIR, config, device, model, dataset)


def test_downstream_classifier(config, device):
    dataset, model, _ = setup_experiment(ROOT_DIR, config, device, train=False)
    run_test(ROOT_DIR, config, device, model, dataset)


def inference(config, device):
    dataset, model, _ = setup_experiment(ROOT_DIR, config, device, train=True)
    run_inference(ROOT_DIR, config, device, model, dataset)


def call_with_configs(config_ls):
    def decorate(run_experiment):
        def wrapper(*args, **kwargs):
            device = setup_device(ROOT_DIR)
            for config in config_ls:
                config = setup_env(ROOT_DIR, config)
                kwargs['device'] = torch.device(device)
                kwargs['config'] = config
                print(f'Starting up experiment on device {device}...')
                run_experiment(**kwargs)
                print(f'Finishing up experiment on device {device}...')
                time.sleep(WAIT_TIME)
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
        os.makedirs(os.path.join(ROOT_DIR, "results"), exist_ok=True)
        os.makedirs(os.path.join(ROOT_DIR, "compare"), exist_ok=True)
        os.makedirs(os.path.join(ROOT_DIR, "configs"), exist_ok=True)
        os.makedirs(os.path.join(ROOT_DIR, "saved_models"), exist_ok=True)
        os.makedirs(os.path.join(ROOT_DIR, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(ROOT_DIR, "tmp"), exist_ok=True)
    except IOError as e:
        traceback.print_exception(*sys.exc_info())
    finally:
        configs = process_arguments(ROOT_DIR)
        call_with_configs(config_ls=configs)(run_experiment)()

if __name__ == "__main__":
    main()
    sys.exit(0)