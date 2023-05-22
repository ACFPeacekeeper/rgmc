import time
import traceback

from utils import *
from tqdm import tqdm
from collections import Counter, defaultdict

# Assign path to current directory
m_path = "/home/pkhunter/Repositories/rmgm/rmgm_code"


def train_model(config):
    device, dataset, model, loss_list_dict, batch_number, optimizer = setup_experiment(m_path, config)

    checkpoint_counter = config['checkpoint'] 
    
    bt_loss = defaultdict(list)
    total_start = time.time()
    tracemalloc.start()
    for epoch in range(config['epochs']):
        print(f'Epoch {epoch}')
        with open(os.path.join(m_path, "results", config['stage'], config['model_out'] + ".txt"), 'a') as file:
            file.write(f'Epoch {epoch}:\n')

        loss_dict = Counter(dict.fromkeys(loss_list_dict.keys(), 0.))

        epoch_start = time.time()
        for batch_idx in tqdm(range(batch_number)):
            # Skip last batch
            batch_end_idx = batch_idx * config['batch_size'] + config['batch_size']
            if batch_end_idx > dataset.dataset_len:
                break
            batch = dict.fromkeys(dataset.dataset.keys())
            for key, value in dataset.dataset.items():
                batch[key] = value[batch_idx * config['batch_size'] : batch_end_idx, :]

            if model.name == 'GMC':
                loss, batch_loss_dict = model.training_step(batch, {"temperature": config['infonce_loss_temperature_scale']}, batch_end_idx - batch_idx * config['batch_size'])
            else:
                x_hat, _ = model(batch)    
                loss, batch_loss_dict = model.loss(batch, x_hat)      

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
            
        wandb.log({**loss_dict})
        epoch_end = time.time()
        print(f'Runtime: {epoch_end - epoch_start} sec')
        with open(os.path.join(m_path, "results", config['stage'], config['model_out'] + ".txt"), 'a') as file:
            file.write(f'- Runtime: {epoch_end - epoch_start} sec\n')
        save_results(m_path, config, device, loss_dict)

        checkpoint_counter -= 1
        if checkpoint_counter == 0:
            print('Saving model checkpoint to file...')
            torch.save(model.state_dict(), os.path.join(m_path, "checkpoints", config['model_out'] + f'_{epoch}.pt'))
            checkpoint_counter = config['checkpoint']

        if device.type == 'cuda':
            torch.cuda.empty_cache()
        tracemalloc.reset_peak()

    tracemalloc.stop()
    total_end = time.time()
    print(f'Total runtime: {total_end - total_start} sec')
    with open(os.path.join(m_path, "results", config['stage'], config['model_out'] + ".txt"), 'a') as file:
        file.write(f'Total runtime: {total_end - total_start} sec\n')
    save_train_results(m_path, config, loss_list_dict, bt_loss)
    save_final_metrics(m_path, config, loss_dict, {key: value[0] for key, value in loss_list_dict.items()}, loss_list_dict)
    torch.save(model.state_dict(), os.path.join(m_path, "saved_models", config['model_out'] + ".pt"))
    json_object = json.dumps(config, indent=4)
    print(config["config_out"])
    with open(os.path.join(m_path, "configs", config['stage'], config["config_out"]), "w") as json_file:
        json_file.write(json_object)

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    wandb.finish()
    return model


def train_downstream_classifier(config):
    if arguments.train_results != 'none':
        with open(os.path.join(m_path, "results", "train_model", arguments.train_results), 'r+') as file:
            lines = file.readlines()
            exclude_modality = [line for line in lines if "Exclude modality" in line][0].split(':')[1].strip()
        
        print(f'Loaded model training configuration from: results/train_model/{arguments.train_results}')
        wandb_dict['model_train_config'] = arguments.train_results
        with open(os.path.join(m_path, "results", arguments.stage, arguments.model_out + ".txt"), 'a') as file:
            file.write(f'Loaded model training configuration from: results/train_model/{arguments.train_results}\n')
    else:
        exclude_modality = arguments.exclude_modality

    device, dataset, model, loss_list_dict, batch_number, optimizer = setup_experiment(m_path, arguments, wandb_dict, exclude_modality, latent_dim=-1, train=True, get_labels=True)
    checkpoint_counter = arguments.checkpoint 
    bt_loss = defaultdict(list)
    total_start = time.time()
    tracemalloc.start()
    for epoch in range(arguments.epochs):
        print(f'Epoch {epoch}')
        with open(os.path.join(m_path, "results", arguments.stage, arguments.model_out + ".txt"), 'a') as file:
            file.write(f'Epoch {epoch}:\n')

        loss_dict = Counter(dict.fromkeys(loss_list_dict.keys(), 0.))

        epoch_start = time.time()

        epoch_preds = [0]*dataset.labels.size(dim=-1)
        for batch_idx in tqdm(range(batch_number)):
            # Skip last batch
            batch_end_idx = batch_idx*arguments.batch_size+arguments.batch_size
            if batch_end_idx > dataset.dataset_len:
                break
            batch = dict.fromkeys(dataset.dataset.keys())
            batch.pop('labels', None)
            for key, value in dataset.dataset.items():
                    batch[key] = value[batch_idx*arguments.batch_size:batch_end_idx, :]
                
            
            batch_labels = dataset.labels[batch_idx*arguments.batch_size:batch_end_idx]

            classification, _, _ = model(batch)
            loss, batch_loss_dict, num_preds = model.loss(classification, batch_labels)

            loss.backward()
            if arguments.optimizer != 'none':
                optimizer.step()
                optimizer.zero_grad()
            
            wandb.log({**batch_loss_dict})
            for key, value in batch_loss_dict.items():
                bt_loss[key].append(float(value))

            loss_dict = loss_dict + batch_loss_dict
            epoch_preds = [sum(x) for x in zip(epoch_preds, num_preds)]

        for key in loss_dict.keys():
            loss_dict[key] = loss_dict[key] / batch_number
            loss_list_dict[key][epoch] = loss_dict[key]

        wandb.log({**loss_dict})
        epoch_end = time.time()
        print(f'Runtime: {epoch_end - epoch_start} sec')
        print(f'Prediction count: {epoch_preds}')
        with open(os.path.join(m_path, "results", arguments.stage, arguments.model_out + ".txt"), 'a') as file:
            file.write(f'- Runtime: {epoch_end - epoch_start} sec\n')
            file.write(f'- Prediction count: {epoch_preds}\n')
        save_results(m_path, arguments, device, loss_dict)

        checkpoint_counter -= 1
        if checkpoint_counter == 0:
            print('Saving model checkpoint to file...')
            torch.save(model.state_dict(), os.path.join(m_path, "checkpoints", f'clf_{arguments.architecture.lower()}_{arguments.dataset.lower()}_{epoch}.pt'))
            checkpoint_counter = arguments.checkpoint

        if device.type == 'cuda':
            torch.cuda.empty_cache()
        tracemalloc.reset_peak()

    tracemalloc.stop()
    total_end = time.time()
    print(f'Total runtime: {total_end - total_start} sec')
    with open(os.path.join(m_path, "results", arguments.stage, arguments.model_out + ".txt"), 'a') as file:
        file.write(f'Total runtime: {total_end - total_start} sec\n')
    save_train_results(m_path, arguments, loss_list_dict, bt_loss)
    save_preds(m_path, arguments, epoch_preds, dataset.labels)
    save_final_metrics(m_path, arguments, loss_dict, {key: value[0] for key, value in loss_list_dict.items()}, loss_list_dict)
    torch.save(model.state_dict(), os.path.join(m_path, "saved_models", arguments.model_out + '.pt'))
    json_object = json.dumps(config, indent=4)
    with open(os.path.join(m_path, "configs", config['stage'], config["config_out"]), "w") as json_file:
        json_file.write(json_object)

    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    wandb.finish()
    return model


def test_model(arguments, device, wandb_dict):
    if arguments.train_results != 'none':
        with open(os.path.join(m_path, "results", "train_model", arguments.train_results), 'r+') as file:
            lines = file.readlines()
            exclude_modality = [line for line in lines if "Exclude modality" in line][0].split(':')[1].strip()
        
        print(f'Loaded model training configuration from: results/train_model/{arguments.train_results}')
        with open(os.path.join(m_path, "results", arguments.model_out + ".txt"), 'a') as file:
            file.write(f'Loaded model training configuration from: results/train_model/{arguments.train_results}\n')
    else:
        exclude_modality = arguments.exclude_modality

    device, dataset, model, loss_dict, _, _ = setup_experiment(m_path, arguments, wandb_dict, exclude_modality, latent_dim=-1, train=False)
    
    tracemalloc.start()
    print('Testing model')
    with open(os.path.join(m_path, "results", arguments.model_out + ".txt"), 'a') as file:
        file.write('Testing model:\n')


    test_start = time.time()
    if model.name == 'GMC':
        dataset_size = list(dataset.values())[0].size(dim=0)
        loss_dict = model.validation_step(dataset, {"temperature": arguments.infonce_temperature}, dataset_size)
    else:
        x_hat, _ = model(dataset)
        _, loss_dict = model.loss(dataset, x_hat)
    test_end = time.time()

    save_final_metrics(arguments, loss_dict)
    print(f'Runtime: {test_end - test_start} sec')
    with open(os.path.join(m_path, "results", arguments.model_out + ".txt"), 'a') as file:
        file.write(f'- Runtime: {test_end - test_start} sec\n')
    save_results(arguments, device, loss_dict)
    tracemalloc.stop()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return

def test_downstream_classifier(arguments, device, wandb_dict):
    if arguments.train_results != 'none':
        results_path = os.path.join(m_path, "results", "train_classifier", arguments.train_results)
        with open(results_path, 'r+') as f:
            lines = f.readlines()
            train_model_file = [line for line in lines if "Loaded model training configuration from" in line][0].split(':')[1].strip()
            with open(os.path.join(m_path, train_model_file)) as file:
                file_lines = file.readlines()
                exclude_modality = [line for line in file_lines if "Exclude modality" in line][0].split(':')[1].strip()

            print(f'Loaded model training configuration from: {train_model_file}')
            f.write(f'Loaded model training configuration from: {train_model_file}\n')
            print(f'Loaded classifier training configuration from: results/train_classifier/{arguments.train_results}')
            f.write(f'Loaded classifier training configuration from: results/train_classifier/{arguments.train_results}\n')
    else:
        exclude_modality = arguments.exclude_modality

    device, dataset, clf, loss_dict, batch_number, optimizer = setup_experiment(m_path, arguments, wandb_dict, exclude_modality, latent_dim=-1, train=False, get_labels=True)

    tracemalloc.start()
    print('Testing classifier')
    with open(os.path.join(m_path, "results", arguments.model_out + ".txt"), 'a') as file:
        file.write('Testing classifier:\n')

    test_start = time.time()
    classification, _, _ = clf(dataset.dataset)
    clf_loss, accuracy, preds = clf.loss(classification, dataset.labels)
    test_end = time.time()

    loss_dict['NLL loss'] = clf_loss
    loss_dict['Accuracy'] = accuracy
    save_final_metrics(os.path.join(m_path, "results", arguments.model_out + ".txt"), loss_dict)
    save_preds(os.path.join(m_path, "results", arguments.model_out + ".txt"), preds, dataset.labels)
    print(f'Runtime: {test_end - test_start} sec')
    with open(os.path.join(m_path, "results", arguments.model_out + ".txt"), 'a') as file:
        file.write(f'- Runtime: {test_end - test_start} sec\n')
    save_results(os.path.join(m_path, "results", arguments.model_out + ".txt"), device, loss_dict)
    tracemalloc.stop()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return

def inference(arguments, device):
    if arguments.train_results != 'none':
        with open(os.path.join(m_path, "results", "train_model", arguments.train_results), 'r+') as file:
            lines = file.readlines()
            exclude_modality = [line for line in lines if "Exclude modality" in line][0].split(':')[1].strip()
        
        print(f'Loaded model training configuration from: {arguments.train_results}')
        with open(os.path.join(m_path, "results", arguments.model_out + ".txt"), 'a') as file:
            file.write(f'Loaded model training configuration from: {arguments.train_results}\n')
    else:
        exclude_modality = arguments.exclude_modality

    if arguments.architecture == 'VAE':
        scales = {'image': arguments.image_scale, 'trajectory': arguments.traj_scale, 'kld beta': arguments.kld_beta}
        model = vae.VAE(arguments.architecture, arguments.latent_dim, device, exclude_modality, scales, arguments.rep_mean, arguments.rep_std, test=True)
    elif arguments.architecture == 'DAE':
        scales = {'image': arguments.image_scale, 'trajectory': arguments.traj_scale}
        model = dae.DAE(arguments.architecture, arguments.latent_dim, device, exclude_modality, scales, test=True)
    elif arguments.architecture == 'GMC':
        model = gmc.MhdGMC(arguments.architecture, exclude_modality, arguments.latent_dim)
    elif arguments.architecture == 'MVAE':
        scales = {'image': arguments.image_scale, 'trajectory': arguments.traj_scale, 'kld beta': arguments.kld_beta}
        model = mvae.MVAE(arguments.architecture, arguments.latent_dim, device, exclude_modality, scales, arguments.rep_mean, arguments.rep_std, arguments.experts_type)

    model.load_state_dict(torch.load(os.path.join(m_path, arguments.path_model)))
    if arguments.train_results != 'none':
        model.set_modalities(arguments.exclude_modality)
    for param in model.parameters():
        param.requires_grad = False

    model.to(device)

    dataset = dataset_setup(arguments, model, device)
    
    tracemalloc.start()
    print('Performing inference')
    with open(os.path.join(m_path, "results", os.path.splitext(os.path.basename(arguments.path_model))[0] + ".txt"), 'a') as file:
        file.write('Performing inference:\n')


    inference_start = time.time()
    x_hat, _ = model(dataset)
    counter = 0
    for idx, (img, recon) in tqdm(enumerate(zip(dataset['image'], x_hat['image'])), total=x_hat['image'].size(dim=0)):
        if counter % arguments.checkpoint == 0: 
            plt.imsave(os.path.join("images", f'{arguments.model_out}_{idx}_orig.png'), torch.reshape(img, (28,28)).detach().clone().cpu())
            plt.imsave(os.path.join("images", f'{arguments.model_out}_{idx}_recon.png'), torch.reshape(recon, (28,28)).detach().clone().cpu())
        counter += 1

    inference_stop = time.time()
    print(f'Runtime: {inference_stop - inference_start} sec')
    with open(os.path.join(m_path, "results", os.path.splitext(os.path.basename(arguments.path_model))[0] + ".txt"), 'a') as file:
        file.write(f'- Runtime: {inference_stop - inference_start} sec\n')
    arguments.model_out = os.path.splitext(os.path.basename(arguments.path_model))[0]
    save_results(arguments, device)
    tracemalloc.stop()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return


def call_with_configs(config_ls):
    def decorate(run_experiment):
        def wrapper(*args, **kwargs):
            for config in config_ls:
                config = setup_env(m_path, config)
                kwargs['config'] = config
                run_experiment(**kwargs)
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
            os.makedirs(os.path.join(m_path, "images"), exist_ok=True)
            inference(config)
    except:
        wandb.finish()
        traceback.print_exception(*sys.exc_info())
        with open(os.path.join(m_path, "experiments_idx.pickle"), 'rb') as idx_pickle:
            idx_dict = pickle.load(idx_pickle)
            idx_dict[config['architecture']][config['dataset']] -= 1
        with open(os.path.join(m_path, "experiments_idx.pickle"), "wb") as idx_pickle:
            pickle.dump(idx_dict, idx_pickle, protocol=pickle.HIGHEST_PROTOCOL)
            
        sys.exit(1)


def main():
    os.makedirs(os.path.join(m_path, "results"), exist_ok=True)
    os.makedirs(os.path.join(m_path, "configs"), exist_ok=True)
    os.makedirs(os.path.join(m_path, "saved_models"), exist_ok=True)
    os.makedirs(os.path.join(m_path, "checkpoints"), exist_ok=True)
    configs = process_arguments(m_path)
    call_with_configs(config_ls=configs)(run_experiment)()
        

if __name__ == "__main__":
    main()
    sys.exit(0)