#!/bin/sh
# Pendulum dataset
# Code adapted from https://github.com/miguelsvasco/gmc

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=10dHskEHRzlpVdX4x6gZJJisxodQtZ43_' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=10dHskEHRzlpVdX4x6gZJJisxodQtZ43_" -O ./train_pendulum_dataset_samples20000_stack2_freq440.0_vel20.0_rec[\'LEFT_BOTTOM\'\,\ \'RIGHT_BOTTOM\'\,\ \'MIDDLE_TOP\'].pt && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=17dbf2pZvFwEN3jbtz7uNQbQP3dqRlVwg' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=17dbf2pZvFwEN3jbtz7uNQbQP3dqRlVwg" -O ./test_pendulum_dataset_samples2000_stack2_freq440.0_vel20.0_rec[\'LEFT_BOTTOM\'\,\ \'RIGHT_BOTTOM\'\,\ \'MIDDLE_TOP\'].pt && rm -rf /tmp/cookies.txt