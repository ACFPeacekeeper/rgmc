# Robust Geometric Multimodal Contrastive Learning Framework


## Setup Conda Environment
To setup the conda environment for the project, you just need to run the following commands in the main directory:
```bash
conda env create -f environment.yml
conda activate RGMC
```
## Setup Project
In order to setup the project, you need to download and prepare each dataset by following the corresponding instructions below.

### Multimodal Handwritten Digits (MHD) Dataset
For the MHD dataset, you need to go into the mhd directory and then run the bash script to download it:
```bash
cd rgmc_code/datasets/mhd
bash download_mhd_dataset.sh
```

### Modified National Institute of Standards and Technology and Street View House Numbers (MNIST-SVHN) Dataset
For the MNIST-SVHN dataset, you need to be in the rgmc_code directory and then run the following python script:
```bash
python download_mnist_svhn_dataset.py
```

## Example commands

### Compare metrics for DAE-based classifier on MHD dataset given different standard deviation values for gaussian noise on image modality
```bash
python main.py compare -a dae -d mhd -s test_classifier --pc noise_std --pp target_modality
```
