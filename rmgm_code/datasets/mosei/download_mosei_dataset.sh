#!/bin/sh
# MOSEI dataset
# Code adapted from https://github.com/miguelsvasco/gmc

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1J65wTEvLR7h7Cjph4LyVWh1Wa7D2sEup' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1J65wTEvLR7h7Cjph4LyVWh1Wa7D2sEup" -O ./mosei_train.pt && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=12N9Maar0SxR2UAh4Oqn4FFAGeraLElbh' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=12N9Maar0SxR2UAh4Oqn4FFAGeraLElbh" -O ./mosei_valid.pt && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1LfogypTvZGnLpo7CUp1_7fbwTGCGHMvg' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1LfogypTvZGnLpo7CUp1_7fbwTGCGHMvg" -O ./mosei_test.pt && rm -rf /tmp/cookies.txt