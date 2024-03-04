# Robust Geometric Multimodal Contrastive Learning Framework


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
They are two different ways you can train and/or test models.
In order to begin a new experiment from the command line, you must choose the architecture, dataset and stage for the experiment:
```bash
python main.py exp --a <architecture> --d <dataset> --s <train_model||train_classifier||test_model||test_classifier>
```
This will begin an experiment with the default hyper-parameters for the given architecture, dataset and stage, but you can also define the values you want for each hyper-parameter in the arguments (e.g., learning rate, batch size, number of epochs). For the full list of hyper-parameters you can tune, [see](https://github.com/MrIceHavoc/rgmc/blob/6d7f73afcb8e87e5dfcb289e43370c49ea07d29c/rgmc_code/utils/command_parser.py#L90).

## Example commands

### Compare metrics for DAE-based classifier on MHD dataset given different standard deviation values for gaussian noise on image modality
```bash
python main.py compare -a dae -d mhd -s test_classifier --pc noise_std --pp target_modality
```
