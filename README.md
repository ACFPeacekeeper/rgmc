# Robust Geometric Multimodal Contrastive Learning Framework
Unsupervised Multimodal DL architectures that are robust to noisy and adversarial data.

## Tech Stack
- [Python Programming Language](https://www.python.org/)
- [PyTorch](https://pytorch.org/)
- [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/)
- [Jupyter](https://jupyter.org/)
- [Anaconda](https://www.anaconda.com/)

### Adapted Code
This project contains code that was adapted from the following repositories:
- [Geometric Multimodal Contrastive Representation Learning](https://github.com/miguelsvasco/gmc)
- [Adversarial Attacks with PyTorch](https://github.com/Harry24k/adversarial-attacks-pytorch)
- [Multimodal Variational Autoencoder](https://github.com/mhw32/multimodal-vae-public)

## Setup Conda Environment
To setup the conda environment for the project, you just need to run the following commands in the main directory:
```bash
conda env create -f environment.yml
conda activate RGMC
```

## Setup Project
In order to setup the project, just change to the rgmc_code directory and then run the python script to download and prepare all datasets:
```bash
cd rgmc_code
python download_datasets.py
```

## Running experiments
There are two different ways you can train and/or test models.

### Command Line 
In order to begin a new experiment from the command line, you must choose the architecture, dataset and stage for the experiment:
```bash
python main.py exp --a <architecture> --d <dataset> --s <train_model||train_classifier||test_model||test_classifier>
```
This will begin an experiment with the default hyper-parameters for the given architecture, dataset and stage, but you can also define the values you want for each hyper-parameter in the arguments (e.g., learning rate, batch size, number of epochs). For the full list of hyper-parameters you can tune, [see](https://github.com/MrIceHavoc/rgmc/blob/6d7f73afcb8e87e5dfcb289e43370c49ea07d29c/rgmc_code/utils/command_parser.py#L90).

### Config File
You can also run several experiments in succession by reading a JSON file with a list of experimental configurations:
```bash
python main.py config --load_config <json_filepath>
```
If instead you want to run multiple experiments with all possible hyper-parameter permutations, you can load the configurations json file with the `--config_permute <json_filepath>` option.

## Compare Experimental Results

For example, to compare metrics for a DAE-based classifier on the MHD dataset given different standard deviation values for gaussian noise on the image modality, you just need to run the following command:

```bash
python main.py compare -a dae -d mhd -s test_classifier --pc noise_std --pp target_modality
```

## Contacts & Questions

For any additional questions, feel free to email `afonso.fernandes[at]tecnico.ulisboa.pt".
