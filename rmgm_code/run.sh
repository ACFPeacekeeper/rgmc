#!/bin/bash
CUR_DIR=$(pwd)
CODE_DIR=${CUR_DIR}/saved_models

declare -a arr=("vae" "dae" "mvae" "gmc")
for model in "${arr[@]}"; do
    python main.py ${model^^} --stage train_classifier --model_out ${model}_mhd_10.pt --path_model saved_models/${model}_mhd_1.pt --noise gaussian --target_modality image
    #python main.py ${model^^} --stage test_classifier --path_model saved_models/${model}_mhd_10.pt --noise gaussian --target_modality image
done
# 4 exclude trajectory, adam, lr=0.001, betas=[0.9, 0.999]
# 5 adam, lr=0.001, betas=[0.9, 0.999]
# 6 exclude image, adam, lr=0.001, betas=[0.9, 0.999]
# 7 batch size 64
# 8 latent dimension 256
# 9 batch size 64, latent dimension 256