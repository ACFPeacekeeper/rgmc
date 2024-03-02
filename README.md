# Robust Geometric Multimodal Contrastive Learning Framework


## Setup Conda Environment
To setup the conda environment for the project, you just need to run the following commands in the main directory:
```bash
conda env create -f environment.yml
conda activate RGMC
```
## Setup Project
In order to setup the project, you first need to go into the datasets directory:
```bash
cd rgmc_code/datasets
```
Afterwards, you need to follow the instructions below for the dataset you want to download and prepare. 

### Multimodal Handwritten Digits (MHD) Dataset
For the MHD dataset, you need to go into the mhd directory and then run the bash script to download it:
```bash
cd mhd
bash download_mhd_dataset.sh
```

### Modified National Institute of Standards and Technology and Street View House Numbers (MNIST-SVHN) Dataset


## Example commands

### Compare metrics for DAE-based classifier on MHD dataset given different standard deviation values for gaussian noise on image modality
```bash
python main.py compare -a dae -d mhd -s test_classifier --pc noise_std --pp target_modality
```

This script setups the datasets, copies the code required to run the GMC framework from the [gmc repository](https://github.com/miguelsvasco/gmc) and defines the [m_path variable](https://github.com/MrIceHavoc/rgmc/blob/main/rgmc_code/main.py#L25). 
