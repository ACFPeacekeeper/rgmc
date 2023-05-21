#!/bin/sh
# MOSI dataset
# Code adapted from https://github.com/miguelsvasco/gmc

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1RdHPXQ7XOx7fj3vO1kVL4yqFch5srYK7' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1RdHPXQ7XOx7fj3vO1kVL4yqFch5srYK7" -O ./mosi_train_a.dt && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1QwDrzycIY6EPeCVcEvpsFULy2Dg8etYG' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1QwDrzycIY6EPeCVcEvpsFULy2Dg8etYG" -O ./mosi_valid_a.dt && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=14nT4DHrbPzknq35TMMiYjgWO7Go-ubj-' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=14nT4DHrbPzknq35TMMiYjgWO7Go-ubj-" -O ./mosi_test_a.dt && rm -rf /tmp/cookies.txt