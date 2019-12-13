#!/bin/bash

printf "\nWelcome to this small assistant for fetching the data. This script does not have any error handling, so if anything breaks, you're on your own. If any of the download links expire, check either the links in the README.md or drop me an email at maximilian@pfundstein.org. Last chance to abort operation :)\n\n"

read -p "Press [Enter] to continue."

printf "\n\nStarting with creating folders.\n\n"

mkdir data
mkdir data/{embeddings,original,preprocessed}

printf "\nDownloading dataset. Around 614MB.\n"

wget https://nextcloud.pfundstein.org/s/Akj6Lam8LyAzjJz/download -O data/original/amazon_review_full_csv.tar.gz

printf "\nExtracting files and doing some file shifting.\n"

tar -zxvf amazon_review_full_csv.tar.gz -C data/original

mv data/original/amazon_review_full_csv/* data/original
rm -rf data/original/amazon_review_full_csv

printf "\nData done. Downloading Word2Vec embeddings. Around 3.6GB.\n"

wget https://nextcloud.pfundstein.org/s/cCqF3LgkiTrZcGJ/download -O data/embeddings/GoogleNews-vectors-negative300.bin

printf "Done. You're good to go to start with the preprocessing! :)"