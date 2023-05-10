# Robust Multimodal Generative Model


## Setup Conda Environment
```bash
conda env create -f environment.yml
conda activate RMGM
```
## Setup Datasets
First, you must clone the repository with the gmc code with the following command:
```bash
git clone https://github.com/miguelsvasco/gmc.git
```

Then execute the following command to setup the datasets required to run the experiments:
```bash
bash setup.sh
```

## Experiments
As indicated in the [gmc repository](https://github.com/miguelsvasco/gmc) instructions, please set up the corresponding local machine path in `ingredients/machine_ingredients.py` file by copying the output of `pwd` to the ingredient file (e.g. for the unsupervised experiment):
```bash
cd unsupervised/
pwd

# Edit unsupervised/ingredients/machine_ingredients.py
@machine_ingredient.config
def machine_config():
    m_path = "copy-output-of-pwd-here"
```

The `setup.sh` already defines this value for use in the RMGM experiments.
