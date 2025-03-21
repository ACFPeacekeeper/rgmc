import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TIMEOUT = 0 # Seconds to wait for user to input notes
WAIT_TIME = 0 # Seconds to wait between sequential experiments

ARCHITECTURES = ['vae', 'dae', 'mvae', 'cmvae', 'cmdvae', 'mdae', 'cmdae', 'gmc', 'dgmc', 'gmcwd', 'rgmc']
DATASETS = ['mhd', 'mnist_svhn', 'mosi', 'mosei']
OPTIMIZERS = ['sgd', 'adam', None]
ADVERSARIAL_ATTACKS = ["gaussian_noise", "fgsm", "pgd", "bim", None]
EXPERTS_FUSION_TYPES = ['poe', 'moe', None]
STAGES = ['train_model', 'train_classifier', 'train_supervised', 'train_rl', 'test_model', 'test_classifier', 'inference']
MODALITIES = {
    'mhd': ['image', 'trajectory', 'sound'],
    'mnist_svhn': ['mnist', 'svhn'],
    'mosei_mosi': ['text', 'audio', 'vision'],
    'pendulum': ['image_t', 'audio_t']
}

SEED = 42
LR_DEFAULT = 0.001
EPOCHS_DEFAULT = 100
BATCH_SIZE_DEFAULT = 64
CHECKPOINT_DEFAULT = 0
LATENT_DIM_DEFAULT = 64
COMMON_DIM_DEFAULT = 64
INFONCE_TEMPERATURE_DEFAULT = 0.1
KLD_BETA_DEFAULT = 0.5
REPARAMETERIZATION_MEAN_DEFAULT = 0.0
REPARAMETERIZATION_STD_DEFAULT = 1.0
EXPERTS_FUSION_DEFAULT = "poe"
POE_EPS_DEFAULT = 1e-8
O3N_LOSS_SCALE_DEFAULT = 1.0
MODEL_TRAIN_NOISE_FACTOR_DEFAULT = 1.0
MOMENTUM_DEFAULT = 0.9
ADAM_BETAS_DEFAULTS = [0.9, 0.999]
NOISE_STD_DEFAULT = 1.0
ADV_EPSILON_DEFAULT = 0.15
ADV_ALPHA_DEFAULT = 25 / 255
ADV_STEPS_DEFAULT = 10
ADV_KAPPA_DEFAULT = 10
ADV_LR_DEFAULT = 0.001
RECON_SCALE_DEFAULTS = {
    'mhd': {'image': 0.5, 'trajectory': 0.5, 'sound': 0.0}, 
    'mnist_svhn': {'mnist': 0.5, 'svhn': 0.5},
    'mosei_mosi': {'text': 1/3, 'audio': 1/3, 'vision': 1/3},
    'pendulum': {'image_t': 0.5, 'audio_t': 0.5}
}

__all__ = [ARCHITECTURES, DATASETS, OPTIMIZERS, ADVERSARIAL_ATTACKS, EXPERTS_FUSION_TYPES, STAGES, MODALITIES, SEED,
           LR_DEFAULT, EPOCHS_DEFAULT, BATCH_SIZE_DEFAULT, CHECKPOINT_DEFAULT, LATENT_DIM_DEFAULT, COMMON_DIM_DEFAULT,
           INFONCE_TEMPERATURE_DEFAULT, KLD_BETA_DEFAULT, REPARAMETERIZATION_MEAN_DEFAULT, REPARAMETERIZATION_STD_DEFAULT,
           EXPERTS_FUSION_DEFAULT, POE_EPS_DEFAULT, O3N_LOSS_SCALE_DEFAULT, MODEL_TRAIN_NOISE_FACTOR_DEFAULT, MOMENTUM_DEFAULT,
           ADAM_BETAS_DEFAULTS, NOISE_STD_DEFAULT, ADV_EPSILON_DEFAULT, ADV_ALPHA_DEFAULT, ADV_STEPS_DEFAULT, ADV_KAPPA_DEFAULT,
           ADV_LR_DEFAULT, RECON_SCALE_DEFAULTS]