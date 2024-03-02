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

## Example commands

### Compare metrics for DAE-based classifier on MHD dataset given different standard deviation values for gaussian noise on image modality
```bash
python main.py compare -a dae -d mhd -s test_classifier --pc noise_std --pp target_modality
```
