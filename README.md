# Robust Geometric Multimodal Contrastive Learning Framework


## Setup Conda Environment
```bash
conda env create -f environment.yml
conda activate RMGM
```
## Setup Project
In order to setup the project, you simply need to execute the following command:
```bash
bash setup.sh
```

## Example commands

### Compare metrics for DAE-based classifier on MHD dataset given different standard deviation values for gaussian noise on image modality
```bash
python main.py compare -a dae -d mhd -s test_classifier --pc noise_std --pp target_modality
```

This script setups the datasets, copies the code required to run the GMC framework from the [gmc repository](https://github.com/miguelsvasco/gmc) and defines the [m_path variable](https://github.com/MrIceHavoc/rmgm/blob/main/rmgm_code/main.py#L25). 
