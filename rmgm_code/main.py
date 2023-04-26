import argparse
import torch
import sys
import os

from architectures import vae, dae
from input_transformations import gaussian_noise, fgsm
from torchvision import transforms
from tqdm import tqdm

torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assign path to current directory
m_path = "<output-of-pwd>"

def process_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="rmgm", description="Program tests the performance and robustness of several generative models with clean and noisy/adversarial samples.")
    parser.add_argument('model_type', choices=['VAE', 'DAE', 'GMC'], help='Model type to be used in the experiment.')
    parser.add_argument('model_file', type=str, help="Filename of the file where the model is to be saved or loaded from.")
    parser.add_argument('-d', '--dataset', type=str, default='MHD', choices=['MHD'], help='Dataset to be used in the experiments.')
    parser.add_argument('-s', '--stage', type=str, default='train', choices=['train', 'test_model', 'test_classifier'], help='Stage(s) of the pipeline to execute in the experiment.')
    parser.add_argument('-e', '--epochs', type=int, default=50, help='Number of epochs to train the model.')
    parser.add_argument('-c', '--checkpoint', type=int, default=10, help='Epoch interval between checkpoints of the model in training.')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Number of samples processed for each model update.')
    parser.add_argument('-l', '--latent_dim', type=int, default=126, help='Dimension of the latent space of the models encodings.')
    parser.add_argument('-n', '--noise', type=str, choices=['gaussian'], help='Apply a type of noise to the model\'s input.')
    parser.add_argument('-a', '--adversarial_attack', '--attack', type=str, choices=['FGSM'], help='Execute an adversarial attack against the model.')
    parser.add_argument('-t', '--target_modality', type=str, choices=['image, trajectory'], help='Define the modality to target with noisy and/or adversarial samples.')
    args = parser.parse_args()

    if args.help:
        parser.print_help()
        sys.exit(0)

    try:
        if args.stage == 'train' or args.stage == 'all':
            if args.epochs < 1:
                raise argparse.ArgumentError("Argument error: number of epochs must be a positive and non-zero integer.")
            elif args.batch_size < 1:
                raise argparse.ArgumentError("Argument error: batch_size value must be a positive and non-zero integer.")
            elif args.latent_dim < 1:
                raise argparse.ArgumentError("Argument error: latent_dim value must be a positive and non-zero integer.")
            elif args.checkpoint < 1:
                raise argparse.ArgumentError("Argument error: checkpoint value must be a positive and non-zero integer.")
            elif args.checkpoint > args.epochs:
                raise argparse.ArgumentError("Argument error: checkpoint value must be smaller than or equal to the number of epochs.")
    except argparse.ArgumentError:
        parser.print_help()
        sys.exit(1)
    return args


def train_model(arguments):
    if arguments.model_type.upper() == 'VAE':
        model = vae.VAE(arguments.latent_dim, device)
    elif arguments.model_type.upper() == 'DAE':
        model = dae.DAE(arguments.latent_dim)

    model.to(device)
    
    #opt = torch.optim.Adam()
    checkpoint_counter = arguments.checkpoint 

    if arguments.noise == "gaussian":
        transform = transforms.Compose([gaussian_noise.GaussianNoise()])
    else:
        transform = None

    if arguments.dataset.upper() == 'MHD':
        dataset = torch.load(os.path.join(cur_dir, "datasets", "mhd", "mhd_train.pt"))

    img_samples = dataset[1]
    traj_samples = dataset[2]
    img_samples = img_samples.to(device)
    traj_samples = traj_samples.to(device)

    epoch_loss = [0.]*arguments.epochs
    for epoch in range(arguments.epochs):
        print(f'Epoch {epoch}')
        for img_sample, traj_sample in tqdm(zip(img_samples, traj_samples), total=len(img_samples)):
            img_sample = torch.flatten(img_sample)
            result = model((img_sample, traj_sample))
            loss = model.loss((img_sample, traj_sample), *result)
            epoch_loss[epoch] += loss
            loss.backward()

        epoch_loss[epoch] /= len(img_samples)
        print(f'Epoch {epoch} loss: {epoch_loss[epoch]}')


        checkpoint_counter -= 1
        if checkpoint_counter == 0:
            print('Saving model checkpoint to file...')
            torch.save(model.state_dict(), os.path.join(cur_dir, "checkpoints", f'{arguments.model_type.lower()}_{arguments.dataset.lower()}_{epoch}.pt'))
            checkpoint_counter = arguments.checkpoint

    print("Saving per epoch loss values to file...")
    with open(os.path.join(cur_dir, f"{arguments.model_type.lower()}_{arguments.dataset.lower()}_results.txt"), 'w') as file:
        file.write(f'{arguments.model_type.upper()} loss per epoch on {arguments.dataset.upper()} dataset\n')
        for idx, loss_value in enumerate(epoch_loss):
            file.write(f'Epoch {idx} loss: {loss_value}\n')

    torch.save(model.state_dict(), os.path.join(cur_dir, "saved_models", arguments.model_file))
    return model

def test_model(arguments):
    #TODO 
    raise NotImplementedError


def test_downstream_classifier(arguments):
    #TODO
    raise NotImplementedError


def main() -> None:
    args = process_arguments()
    print(f'Model type: {args.model_type}')
    print(f'Pipeline stage: {args.stage}')
    if args.stage == 'train':
        os.makedirs(os.path.join(cur_dir, "saved_models"), exist_ok=True)
        os.makedirs(os.path.join(cur_dir, "checkpoints"), exist_ok=True)
        train_model(args)
    elif args.stage == 'test_model':
        os.makedirs(os.path.join(cur_dir, "model_test_results"))
        test_model(args)
    elif args.stage == 'test_classifier':
        os.makedirs(os.path.join(cur_dir, "classifier_test_results"))
        test_downstream_classifier(args)

if __name__ == "__main__":
    main()