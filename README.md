# Robust Multimodal Generative Model


## Setup Conda Environment
```bash
conda env create -f environment.yml
conda activate RMGM
```
## Setup Datasets
Run the following command to setup the datasets required to run the experiments:
```bash
bash setup.sh
```

## Experiments
In every GMC experiment, please set up the corresponding local machine path in `ingredients/machine_ingredients.py` file by copying the output of `pwd` to the ingredient file (e.g. for the unsupervised experiment):
```bash
cd unsupervised/
pwd

# Edit unsupervised/ingredients/machine_ingredients.py
@machine_ingredient.config
def machine_config():
    m_path = "copy-output-of-pwd-here"
```

The `setup.sh` already defines this value for use in the RMGM experiments.
