#!/bin/sh
# MOSI dataset
# Code adapted from https://github.com/miguelsvasco/gmc

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1obLxMDukCZsu9sTEifwG0JD4gLPLJ5VT' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1obLxMDukCZsu9sTEifwG0JD4gLPLJ5VT" -O ./mosi_train.dt && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1vvDhAZ5V6yrowPy26ND3i13QRGE8ybUH' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1vvDhAZ5V6yrowPy26ND3i13QRGE8ybUH" -O ./mosi_valid.pt && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1GCpWQM8xhL72zt6NM1Om4bJCvD-bhpuk' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1GCpWQM8xhL72zt6NM1Om4bJCvD-bhpuk" -O ./mosi_test.pt && rm -rf /tmp/cookies.txt