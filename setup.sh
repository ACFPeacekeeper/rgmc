#!/bin/bash
CUR_DIR=$(pwd)
DATA_DIR=${CUR_DIR}/rmgm_code/datasets
GMC_DIR=${CUR_DIR}/rmgm_code/gmc/gmc_code
MODELS_DIR=${CUR_DIR}/rmgm_code/architectures

if ! [ -d "$GMC_DIR" ]; then
    echo "Cloning GMC..."
    git clone "https://github.com/miguelsvasco/gmc.git" ${CUR_DIR}/rmgm_code/gmc
    echo "Completed! $GMC_DIR directory clone from repository."
fi

FILE=${MODELS_DIR}/gmc_networks.py
if ! [ -f "$FILE" ]; then
    echo "Copying gmc_networks.py to ${MODELS_DIR} directory..."
    cp ${GMC_DIR}/unsupervised/architectures/models/gmc_networks.py ${FILE}
    echo "Completed! ${FILE} created."
fi

FILE=${MODELS_DIR}/gmc.py
if ! [ -f "$FILE" ]; then
    echo "Copying gmc.py to ${MODELS_DIR} directory..."
    cp ${GMC_DIR}/unsupervised/architectures/models/gmc.py ${FILE}
    sed -i -e "s|^from gmc_code.*$|from architectures.gmc_networks import \*\n|g" ${MODELS_DIR}/gmc.py
    echo "Completed! ${FILE} created."
fi


if ! [ -d "$DATA_DIR" ]; then
    echo "Setting up $DATA_DIR directory..."
    mkdir ${DATA_DIR}
    touch ${DATA_DIR}/__init__.py
    echo "Completed! $DATA_DIR directory created."
fi

COMMANDS='#!/bin/bash\nTARGET_DIR\=\$(pwd)\nTARGET_DIR=\${TARGET_DIR}/rmgm_code/gmc/gmc_code\n'
declare -a GMC_TASKS=("rl" "supervised" "unsupervised")
declare -a RMGM_DATASETS=("pendulum"  "mosi_mosei" "mhd")
declare -a FILE_LIST=("train_pendulum_dataset_samples20000_stack2_freq440.0_vel20.0_rec['LEFT_BOTTOM', 'RIGHT_BOTTOM', 'MIDDLE_TOP'].pt;test_pendulum_dataset_samples2000_stack2_freq440.0_vel20.0_rec['LEFT_BOTTOM', 'RIGHT_BOTTOM', 'MIDDLE_TOP'].pt;" "mosei_test_a.dt;mosi_test_a.dt;mosei_valid_a.dt;mosei_train_a.dt;mosi_train_a.dt;mosi_valid_a.dt;" "mhd_test.pt;mhd_train.pt;")

for idx in "${!GMC_TASKS[@]}"; do
    idx_file_list=${FILE_LIST[$idx]}
    
    task=${GMC_TASKS[$idx]}
    gmc_dataset_files=`find ${GMC_DIR}/${task}/dataset -name "*.[dp]t" -type f -printf "%f;"`
    file_diff=$(diff <(echo "$idx_file_list") <(echo "$gmc_dataset_files"))

    if [ "$file_diff" != "" ]; then
        download_script=${CUR_DIR}/rmgm_code/gmc/gmc_code/download_${task}_dataset.sh

        if IFS= read -r firstline < "$download_script" && ! [[ $firstline == '#!/bin/bash' ]]; then
            echo "Making ${download_script} executable..."
            sed -i "/${firstline}/i ${COMMANDS}" $download_script
            sed -i -e 's/-O ./-O ${TARGET_DIR}/g' $download_script
            chmod u+x $download_script
            echo "Completed! The script ${download_script} is now executable."
        fi

        echo "Downloading data to ${GMC_DIR}/${task}/dataset..."
        bash $download_script
        echo "Completed! Downloaded the following files: ${idx_file_list}."
    fi

    dataset=${RMGM_DATASETS[$idx]}
    rmgm_dataset_files=`find ${DATA_DIR}/${dataset} -name "*.[dp]t" -type f -printf "%f;"`
    file_diff=$(diff <(echo "$idx_file_list") <(echo "$rmgm_dataset_files"))

    if [ "$file_diff" != "" ]; then
        echo IDX FILE LIST $idx_file_list
        echo RMGM DATASET FILES $rmgm_dataset_files

        echo "Setting up the ${DATA_DIR}/${dataset} directory..."
        cp -r ${GMC_DIR}/${task}/dataset ${DATA_DIR}/${dataset}
        #ln -s ${GMC_DIR}/${task}/dataset/*.[dp]t ${DATA_DIR}/${dataset}/
        touch ${DATA_DIR}/${dataset}/__init__.py
        echo "Completed! Copied the files in ${GMC_DIR}/${task}/dataset to ${DATA_DIR}/${dataset}."
    fi

done

sed -i -e "s|^m_path.*$|m_path = \"${CUR_DIR}/rmgm_code\"|g" ${CUR_DIR}/rmgm_code/main.py

rm -rf ${CUR_DIR}/rmgm_code/gmc

echo "Setup completed!"