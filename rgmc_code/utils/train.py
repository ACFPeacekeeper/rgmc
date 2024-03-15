import wandb
import tracemalloc

from time import time
from tqdm import tqdm
from json import dumps
from os.path import join
from torch import isnan, save, cuda
from torch.utils.data import DataLoader
from collections import Counter, defaultdict
from utils.logger import save_epoch_results, save_train_results


def nan_hook(self, input, output):
    if isinstance(output, dict):
        outputs = [value for value in output.values()]
    if not isinstance(output, tuple):
        outputs = [output]
    else:
        outputs = output

    for i, out in enumerate(outputs):
        if isinstance(out, dict):
            out = list(out.values())
            
        if isinstance(out, list):
            for value in out:
                nan_mask = isnan(value)    
                if nan_mask.any():
                    print("In", self.__class__.__name__)
                    raise ValueError(f"Found NAN in output.")# {i}") at indices: "), nan_mask.nonzero(), "where:", out[nan_mask.nonzero()[:, 0].unique(sorted=True)])
        else:
            nan_mask = isnan(out)
            if nan_mask.any():
                print("In", self.__class__.__name__)
                raise ValueError(f"Found NAN in output.")# {i} at indices: ), nan_mask.nonzero(), "where:", out[nan_mask.nonzero()[:, 0].unique(sorted=True)])

def run_train_epoch(m_path, epoch, config, device, model, train_set, train_losses, checkpoint_counter, optimizer=None):
    print(f'Epoch {epoch}')
    print('Training:')
    with open(join(m_path, "results", config['stage'], config['model_out'] + ".txt"), 'a') as file:
        file.write(f'Epoch {epoch}\n')
        file.write('Training:\n')

    loss_dict = Counter(dict.fromkeys(train_losses.keys(), 0.))
    train_loader = iter(DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, drop_last=True))
    train_bnumber = len(train_loader)
    run_start = time()
    for batch_feats, batch_labels in tqdm(train_loader, total=train_bnumber):
        if config['optimizer'] is not None:
            optimizer.zero_grad()

        loss, batch_loss_dict = model.training_step(batch_feats, batch_labels)
        loss_dict = loss_dict + batch_loss_dict

        loss.backward()
        if config['optimizer'] is not None:
            optimizer.step()

        if 'wandb' in config and config['wandb']:
            wandb.log({**batch_loss_dict})

    for key, value in loss_dict.items():
        loss_dict[key] = value / train_bnumber
        train_losses[key].append(float(loss_dict[key]))

    run_end = time()
    save_epoch_results(m_path, config, device, run_end - run_start, loss_dict)

    checkpoint_counter -= 1
    if checkpoint_counter == 0:
        print('Saving model checkpoint to file...')
        save(model.state_dict(), join(m_path, "checkpoints", config['model_out'] + f'_{epoch}.pt'))
        checkpoint_counter = config['checkpoint']

    tracemalloc.reset_peak()
    if device.type == 'cuda':
        cuda.empty_cache()

    return model, train_losses, checkpoint_counter, optimizer

def run_training(m_path, config, device, dataset, model, optimizer):
    checkpoint_counter = config['checkpoint']
    for module in model.modules():
        module.register_forward_hook(nan_hook)
    
    train_losses = defaultdict(list)
    total_start = time()
    tracemalloc.start()
    for epoch in range(config['epochs']):
        model, train_losses, checkpoint_counter, optimizer = run_train_epoch(m_path, epoch, config, device, model, dataset, train_losses, checkpoint_counter, optimizer)

    tracemalloc.stop()
    total_end = time()
    print("Train resume:")
    print(f'- Total runtime: {total_end - total_start} sec')
    with open(join(m_path, "results", config['stage'], config['model_out'] + ".txt"), 'a') as file:
        file.write(f'Train resume:\n')
        file.write(f'- Total runtime: {total_end - total_start} sec\n')
    save_train_results(m_path, config, train_losses)

    json_object = dumps(config, indent=4)
    with open(join(m_path, "configs", config['stage'], config["model_out"] + '.json'), "w") as json_file:
        json_file.write(json_object)

    # Save model
    save(model.state_dict(), join(m_path, "saved_models", config['model_out'] + ".pt"))
    if device.type == 'cuda':
        cuda.empty_cache()

    if 'wandb' in config and config['wandb']:
        wandb.finish()
    return model