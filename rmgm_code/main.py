import time
import traceback

from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import Counter, defaultdict

from utils.logger import *
from utils.config_parser import *

# Assign path to current directory
m_path = "/home/pkhunter/Repositories/rmgm/rmgm_code"

WAIT_TIME = 0 # Seconds to wait between sequential experiments

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
                nan_mask = torch.isnan(value)    
                if nan_mask.any():
                    print("In", self.__class__.__name__)
                    raise ValueError(f"Found NAN in output {i} at indices: ", nan_mask.nonzero(), "where:", out[nan_mask.nonzero()[:, 0].unique(sorted=True)])
        else:
            nan_mask = torch.isnan(out)
            if nan_mask.any():
                print("In", self.__class__.__name__)
                raise ValueError(f"Found NAN in output {i} at indices: ", nan_mask.nonzero(), "where:", out[nan_mask.nonzero()[:, 0].unique(sorted=True)])


def run_train_epoch(epoch, config, device, model, dataset, batch_number, loss_list_dict, bt_loss, checkpoint_counter, optimizer=None):
    print(f'Epoch {epoch}')
    with open(os.path.join(m_path, "results", config['stage'], config['model_out'] + ".txt"), 'a') as file:
        file.write(f'Epoch {epoch}:\n')

    loss_dict = Counter(dict.fromkeys(loss_list_dict.keys(), 0.))
    dataloader = iter(DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True))

    epoch_start = time.time()
    for batch_feats, batch_labels in tqdm(dataloader, total=batch_number):
        loss, batch_loss_dict = model.training_step(batch_feats, batch_labels)

        loss.backward()
        if config['optimizer'] is not None:
            optimizer.step()
            optimizer.zero_grad()

        wandb.log({**batch_loss_dict})

        for key, value in batch_loss_dict.items():
            bt_loss[key].append(float(value))

        loss_dict = loss_dict + batch_loss_dict
    
    for key, value in loss_dict.items():
        loss_dict[key] = value / batch_number
        loss_list_dict[key][epoch] = loss_dict[key]
        
    epoch_end = time.time()
    wandb.log({**loss_dict})
    print(f'Runtime: {epoch_end - epoch_start} sec')
    with open(os.path.join(m_path, "results", config['stage'], config['model_out'] + ".txt"), 'a') as file:
        file.write(f'- Runtime: {epoch_end - epoch_start} sec\n')
    save_epoch_results(m_path, config, device, loss_dict)

    checkpoint_counter -= 1
    if checkpoint_counter == 0:
        print('Saving model checkpoint to file...')
        torch.save(model.state_dict(), os.path.join(m_path, "checkpoints", config['model_out'] + f'_{epoch}.pt'))
        checkpoint_counter = config['checkpoint']

    tracemalloc.reset_peak()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return model, loss_list_dict, bt_loss, checkpoint_counter, optimizer

def run_test(config, device, model, dataset, batch_number, loss_list_dict):
    dataloader = iter(DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True))
    tracemalloc.start()
    test_start = time.time()
    for batch_feats, batch_labels in tqdm(dataloader, total=batch_number):
        _, batch_loss_dict = model.validation_step(batch_feats, batch_labels)

        for key in loss_list_dict.keys():
            loss_list_dict[key].append(float(batch_loss_dict[key]))

    test_end = time.time()
    wandb.log({**loss_list_dict})

    test_end = time.time()
    tracemalloc.stop()
    wandb.log({**loss_list_dict})
    print(f'Total runtime: {test_end - test_start} sec')
    with open(os.path.join(m_path, "results", config['stage'], config['model_out'] + ".txt"), 'a') as file:
        file.write(f'Total runtime: {test_end - test_end} sec\n')

    save_test_results(m_path, config, loss_list_dict)
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return loss_list_dict


def train_model(config):
    device, dataset, model, loss_list_dict, batch_number, optimizer = setup_experiment(m_path, config, train=True)
    checkpoint_counter = config['checkpoint'] 
    for module in model.modules():
        module.register_forward_hook(nan_hook)

    bt_loss = defaultdict(list)
    total_start = time.time()
    tracemalloc.start()
    for epoch in range(config['epochs']):
        model, loss_list_dict, bt_loss, checkpoint_counter, optimizer = run_train_epoch(epoch, config, device, model, dataset, batch_number, loss_list_dict, bt_loss, checkpoint_counter, optimizer)

    tracemalloc.stop()
    total_end = time.time()
    print(f'Total runtime: {total_end - total_start} sec')
    with open(os.path.join(m_path, "results", config['stage'], config['model_out'] + ".txt"), 'a') as file:
        file.write(f'Total runtime: {total_end - total_start} sec\n')
    save_train_results(m_path, config, loss_list_dict, bt_loss)
    torch.save(model.state_dict(), os.path.join(m_path, "saved_models", config['model_out'] + ".pt"))
    json_object = json.dumps(config, indent=4)
    with open(os.path.join(m_path, "configs", config['stage'], config["model_out"] + '.json'), "w") as json_file:
        json_file.write(json_object)

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    wandb.finish()
    return model


def train_downstream_classifier(config):
    device, dataset, model, loss_list_dict, batch_number, optimizer = setup_experiment(m_path, config, train=True)
    checkpoint_counter = config['checkpoint'] 
    for module in model.modules():
        module.register_forward_hook(nan_hook)

    bt_loss = defaultdict(list)
    total_start = time.time()
    tracemalloc.start()
    for epoch in range(config['epochs']):
        model, loss_list_dict, bt_loss, checkpoint_counter, optimizer = run_train_epoch(epoch, config, device, model, dataset, batch_number, loss_list_dict, bt_loss, checkpoint_counter, optimizer)

    tracemalloc.stop()
    total_end = time.time()
    print(f'Total runtime: {total_end - total_start} sec')
    with open(os.path.join(m_path, "results", config['stage'], config['model_out'] + ".txt"), 'a') as file:
        file.write(f'Total runtime: {total_end - total_start} sec\n')
    save_train_results(m_path, config, loss_list_dict, bt_loss)
    torch.save(model.state_dict(), os.path.join(m_path, "saved_models", config['model_out'] + '.pt'))
    json_object = json.dumps(config, indent=4)
    with open(os.path.join(m_path, "configs", config['stage'], config["model_out"] + '.json'), "w") as json_file:
        json_file.write(json_object)

    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    wandb.finish()
    return model


def test_model(config):
    device, dataset, model, loss_dict, batch_number, _ = setup_experiment(m_path, config, train=False)
    tracemalloc.start()
    loss_dict = run_test(config, device, model, dataset, batch_number, loss_dict)
    wandb.finish()
    return

def test_downstream_classifier(config):
    device, dataset, model, loss_dict, batch_number, _ = setup_experiment(m_path, config, train=False)
    loss_dict = run_test(config, device, model, dataset, batch_number, loss_dict)
    wandb.finish()
    return

def inference(config):
    device, dataset, model, _, _, _ = setup_experiment(m_path, config, train=False)
    
    tracemalloc.start()
    print('Performing inference')
    with open(os.path.join(m_path, "results", os.path.splitext(os.path.basename(config['path_model']))[0] + ".txt"), 'a') as file:
        file.write('Performing inference:\n')


    inference_start = time.time()
    x_hat, _ = model(dataset)
    counter = 0
    for idx, (img, recon) in tqdm(enumerate(zip(dataset['image'], x_hat['image'])), total=x_hat['image'].size(dim=0)):
        if counter % config['checkpoint'] == 0: 
            plt.imsave(os.path.join("images", config['model_out'] + f'_{idx}_orig.png'), torch.reshape(img, (28,28)).detach().clone().cpu())
            plt.imsave(os.path.join("images", config['model_out'] + f'_{idx}_recon.png'), torch.reshape(recon, (28,28)).detach().clone().cpu())
        counter += 1

    inference_stop = time.time()
    print(f'Runtime: {inference_stop - inference_start} sec')
    with open(os.path.join(m_path, "results", os.path.splitext(os.path.basename(config['path_model']))[0] + ".txt"), 'a') as file:
        file.write(f'- Runtime: {inference_stop - inference_start} sec\n')
    config['model_out'] = os.path.splitext(os.path.basename(config['path_model']))[0]
    save_results(config, device)
    tracemalloc.stop()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    wandb.finish()
    return


def call_with_configs(config_ls):
    def decorate(run_experiment):
        def wrapper(*args, **kwargs):
            for config in config_ls:
                try:
                    config = setup_env(m_path, config)
                    kwargs['config'] = config
                    run_experiment(**kwargs)
                except Exception as e:
                    traceback.print_exception(*sys.exc_info())
                finally:
                    print('Finishing up run...')
                    time.sleep(WAIT_TIME)
                    continue
        return wrapper
    return decorate


def run_experiment(**kwargs):
    config = kwargs['config']

    try:
        if config['stage'] == 'train_model':
            train_model(config)
        elif config['stage'] == 'train_classifier':
            train_downstream_classifier(config)
        elif config['stage'] == 'test_model':
            test_model(config)
        elif config['stage'] == 'test_classifier':
            test_downstream_classifier(config)
        elif config['stage'] == 'inference':
            try:
                os.makedirs(os.path.join(m_path, "images"), exist_ok=True)
            except IOError as e:
                traceback.print_exception(*sys.exc_info())
            finally:
                inference(config)
    except:
        wandb.finish(exit_code=1)
        traceback.print_exception(*sys.exc_info())

def main():
    try:
        os.makedirs(os.path.join(m_path, "results"), exist_ok=True)
        os.makedirs(os.path.join(m_path, "configs"), exist_ok=True)
        os.makedirs(os.path.join(m_path, "saved_models"), exist_ok=True)
        os.makedirs(os.path.join(m_path, "checkpoints"), exist_ok=True)
    except IOError as e:
        traceback.print_exception(*sys.exc_info())
    finally:
        configs = process_arguments(m_path)
        call_with_configs(config_ls=configs)(run_experiment)()

if __name__ == "__main__":
    main()
    sys.exit(0)